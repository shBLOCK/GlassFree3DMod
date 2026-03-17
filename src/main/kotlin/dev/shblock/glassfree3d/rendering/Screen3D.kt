package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.pipeline.MainTarget
import com.mojang.blaze3d.platform.GlStateManager
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.*
import dev.shblock.glassfree3d.ducks.LevelRendererAccessor
import dev.shblock.glassfree3d.utils.*
import net.minecraft.client.Camera
import net.minecraft.client.Minecraft
import net.minecraft.client.multiplayer.ClientLevel
import net.minecraft.client.renderer.LevelRenderer
import net.minecraft.client.renderer.Rect2i
import net.minecraft.client.renderer.RenderBuffers
import net.minecraft.client.renderer.ShaderInstance
import net.minecraft.core.BlockPos
import net.minecraft.resources.ResourceKey
import net.minecraft.world.level.ChunkPos
import net.minecraft.world.level.Level
import net.minecraft.world.level.block.state.BlockState
import net.neoforged.bus.api.SubscribeEvent
import net.neoforged.fml.common.EventBusSubscriber
import net.neoforged.neoforge.client.event.ClientTickEvent
import org.joml.Matrix4d
import org.joml.Matrix4f
import org.joml.Quaterniond
import org.joml.Quaternionf
import org.joml.Vector2d
import org.joml.Vector3d
import org.lwjgl.glfw.GLFW.glfwMakeContextCurrent
import org.lwjgl.opengl.GL33.*
import java.lang.Runtime
import java.util.*

class Screen3D(
    val window: ModWindow,
    var viewport: Rect2i,

    var realPose: Pose = Pose(),
    var realSize: Vector2d = Vector2d(1.0),
    var virtualPose: Pose = Pose(),
    var virtualSize: Vector2d = Vector2d(1.0),

    var realCameraPos: Vector3d = Vector3d(0.0, 0.0, 1.0),

    var zNear: Double = 0.05,
    var clipAtScreenPlane: Boolean = true,
) {
    class Pose(
        var pos: Vector3d = Vector3d(),
        var orientation: Quaterniond = Quaterniond(),
        var scale: Double = 1.0,
        var parent: Pose? = null
    ) {
        fun transform(vec: Vector3d): Vector3d =
            orientation.transform(vec, Vector3d()).mul(scale).add(pos)

        fun global(): Pose {
            val gParent = parent?.global() ?: return this.copy()
            return Pose(
                pos = gParent.transform(pos),
                orientation = gParent.orientation.mul(orientation, Quaterniond()),
                scale = gParent.scale * scale
            )
        }

        fun copy() = Pose(
            pos = Vector3d(pos),
            orientation = Quaterniond(orientation),
            scale = scale,
            parent = parent
        )
    }

    val framebuffer = MainTarget(viewport.width, viewport.height)

    var virtualCameraPos = Vector3d(0.0, 0.0, 1.0)
        private set

    private val virtualCamera = Camera()
    private var frustumMatrix = Matrix4d()
    private var projectionMatrix = Matrix4d()

    init {
        Manager.newScreen(this)
    }

    private fun updateProjectionAndCamera(): Boolean {
        val gRealPose = realPose.global()
        val gVirtualPose = virtualPose.global()
        val gRealSize = realSize.mul(gRealPose.scale, Vector2d())
        val gVirtualSize = virtualSize.mul(gVirtualPose.scale, Vector2d())

        val scale = gVirtualSize.div(gRealSize, Vector2d())
        val scale3d = Vector3d(scale, (scale.x + scale.y) / 2.0)
        val localRealCameraPos = gRealPose.orientation.transformInverse(realCameraPos - gRealPose.pos)
        val localVirtualCameraPos = localRealCameraPos.mul(scale3d, Vector3d())
        if (localVirtualCameraPos.z <= 0.0) return false // camera is behind screen
        virtualCameraPos = gVirtualPose.orientation.transform(localVirtualCameraPos, Vector3d()) + gVirtualPose.pos

        if (clipAtScreenPlane) {
            zNear = localVirtualCameraPos.z
        }
        virtualCamera.initialized = true
        virtualCamera.position = virtualCameraPos.toVec3()
        virtualCamera.rotation.set(gVirtualPose.orientation)
        frustumMatrix.rotation(Quaternionf(gVirtualPose.orientation.conjugate(Quaterniond())))
        val halfVirtualSize = gVirtualSize.div(2.0, Vector2d())
        val left = (-localVirtualCameraPos.x - halfVirtualSize.x) / localVirtualCameraPos.z * zNear
        val right = (-localVirtualCameraPos.x + halfVirtualSize.x) / localVirtualCameraPos.z * zNear
        val bottom = (-localVirtualCameraPos.y - halfVirtualSize.y) / localVirtualCameraPos.z * zNear
        val top = (-localVirtualCameraPos.y + halfVirtualSize.y) / localVirtualCameraPos.z * zNear
        projectionMatrix.setFrustum(
            left, right, bottom, top,
            zNear, MC.gameRenderer.depthFar.toDouble()
        )
        return true
    }

    private fun render() {
        RenderSystem.assertOnRenderThread()

        if (!updateProjectionAndCamera()) return

        MiscUtils.withMainRenderTarget(framebuffer) {
            framebuffer.resizeLazy(viewport.width, viewport.height)

            framebuffer.bindWrite(true)

            val levelRenderer = Manager.getLevelRenderer(MC.level!!.dimension())
            virtualCamera.level = levelRenderer.level!!
            virtualCamera.entity = MC.player!!

            val frustumMatrixF = Matrix4f(frustumMatrix)
            val projectionMatrixF = Matrix4f(projectionMatrix)

            RenderSystem.setProjectionMatrix(projectionMatrixF, VertexSorting.DISTANCE_TO_ORIGIN)

            if (!MC.isPaused) {
                levelRenderer.tickRain(virtualCamera)
            }
            
            levelRenderer.prepareCullFrustum(
                virtualCamera.position,
                frustumMatrixF,
                projectionMatrixF
            )
            levelRenderer.renderLevel(
                MC.timer,
                true,
                virtualCamera,
                MC.gameRenderer,
                MC.gameRenderer.lightTexture(),
                frustumMatrixF,
                projectionMatrixF
            )

            framebuffer.unbindWrite()
        }
    }

    private fun blit() {
        window.blitFramebuffer(framebuffer, viewport, flip = false)
    }

    @Suppress("FunctionName")
    @EventBusSubscriber
    object Manager {
        private val screens = mutableListOf<Screen3D>()
        private val windows = mutableSetOf<ModWindow>()
        private val levelRenderers = mutableMapOf<ResourceKey<Level>, LevelRenderer>()
        val afterRenderAll = mutableListOf<() -> Unit>()

        internal fun newScreen(screen: Screen3D) {
            screens += screen
            windows += screen.window
        }

        internal fun getLevelRenderer(dim: ResourceKey<Level>): LevelRenderer {
            return levelRenderers.getOrPut(dim) {
                LevelRenderer(
                    MC,
                    MC.entityRenderDispatcher,
                    MC.blockEntityRenderDispatcher,
                    RenderBuffers(Runtime.getRuntime().availableProcessors())
                ).apply {
                    (this as LevelRendererAccessor).gf_setDisableFrustumCulling(true)

                    setLevel(MC.level) // TODO: actually handle non-current levels
                }
            }
        }
        
        @SubscribeEvent
        fun onPostClientTick(event: ClientTickEvent.Post) {
            if (!MC.isPaused) {
                levelRenderers.values.forEach { it.tick() }
            }
        }

        internal fun renderAll() {
//            levelRenderers.forEach { level, renderer -> renderer.visibleSections.clear() }
            screens.forEach { it.render() }

            RenderSystem.replayQueue()
            Tesselator.getInstance().clear()

            screens.groupBy(Screen3D::window).forEach { (window, windowScreens) ->
                window.makeCurrent()
                windowScreens.forEach { it.blit() }
                window.endFrame()
            }

            glfwMakeContextCurrent(MC.window.window)

            afterRenderAll.forEach { it() }
        }

        internal fun LR_onChunkLoaded(dim: ResourceKey<Level>, chunkPos: ChunkPos) {
            levelRenderers[dim]?.apply {
                onChunkLoaded(chunkPos)
            }
        }

        internal fun LR_blockChanged(
            level: ClientLevel,
            pos: BlockPos,
            oldState: BlockState,
            newState: BlockState,
            flags: Int
        ) {
            levelRenderers[level.dimension()]?.apply {
                blockChanged(level, pos, oldState, newState, flags)
            }
        }

        internal fun LR_setBlockDirty(
            dim: ResourceKey<Level>,
            blockPos: BlockPos,
            oldState: BlockState,
            newState: BlockState
        ) {
            levelRenderers[dim]?.apply {
                setBlockDirty(blockPos, oldState, newState)
            }
        }

        internal fun LR_setSectionDirtyWithNeighbors(
            dim: ResourceKey<Level>,
            sectionX: Int,
            sectionY: Int,
            sectionZ: Int
        ) {
            levelRenderers[dim]?.apply {
                setSectionDirtyWithNeighbors(sectionX, sectionY, sectionZ)
            }
        }

        internal fun LR_destroyBlockProgress(dim: ResourceKey<Level>, breakerId: Int, pos: BlockPos, progress: Int) {
            levelRenderers[dim]?.apply {
                destroyBlockProgress(breakerId, pos, progress)
            }
        }

        internal fun LR_globalLevelEvent(dim: ResourceKey<Level>, id: Int, pos: BlockPos, data: Int) {
            levelRenderers[dim]?.apply {
                globalLevelEvent(id, pos, data)
            }
        }

        internal fun LR_levelEvent(dim: ResourceKey<Level>, type: Int, pos: BlockPos, data: Int) {
            levelRenderers[dim]?.apply {
                levelEvent(type, pos, data)
            }
        }
    }
}