package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.pipeline.MainTarget
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.*
import dev.shblock.glassfree3d.ducks.LevelRendererAccessor
import dev.shblock.glassfree3d.utils.*
import net.minecraft.client.Camera
import net.minecraft.client.multiplayer.ClientLevel
import net.minecraft.client.renderer.LevelRenderer
import net.minecraft.client.renderer.Rect2i
import net.minecraft.client.renderer.RenderBuffers
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
import java.lang.Runtime

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
        fun transform(vec: Vector3d): Vector3d = transformAffine(vec).add(pos)

        fun transformAffine(vec: Vector3d): Vector3d = orientation.transform(vec, Vector3d()).mul(scale)

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
    private var localVirtualCameraPos = Vector3d(0.0, 0.0, 1.0)
    var zScreen = 0.0
        private set

    private val virtualCamera = Camera()
    var frustumMatrix = Matrix4d()
        private set
    var projectionMatrix = Matrix4d()
        private set

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
        localVirtualCameraPos = localRealCameraPos.mul(scale3d, Vector3d())
        if (localVirtualCameraPos.z <= 0.0) return false // camera is behind screen
        virtualCameraPos = gVirtualPose.orientation.transform(localVirtualCameraPos, Vector3d()) + gVirtualPose.pos

        zScreen = localVirtualCameraPos.z
        if (clipAtScreenPlane) {
            zNear = zScreen
        }
        val zNear = zNear * gVirtualPose.scale
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

    /**
     * @return ray direction vector, normalized so that its projection length on the camera's actual imaginary center axis (not the off-center axis) is one (scaled by [virtualPose]).
     */
    fun unprojectVirtualScreenLocal(ndc: Vector2d): Vector3d {
        val ray = Vector3d(localVirtualCameraPos).mul(-1.0).add(
            Vector3d(Vector2d(ndc).mul(virtualSize).mul(0.5), 0.0)
        )
        ray.div(-ray.z) // "normalize"
        return ray
    }

    fun unprojectVirtualScreenGlobal(ndc: Vector2d) =
        virtualPose.global().transformAffine(unprojectVirtualScreenLocal(ndc))
    
    val afterRender = arrayListOf<(Screen3D) -> Unit>()

    private fun render() {
        RenderSystem.assertOnRenderThread()

        val partialTick = MC.timer.getGameTimeDeltaPartialTick(true)
        
        virtualCamera.setup(
            MC.level!!,
            MC.player!!,
//            true,
            false, // don't render player
            false,
            partialTick
        )

        if (!updateProjectionAndCamera()) return

        if (viewport.width == 0 || viewport.height == 0) return
        
        MiscUtils.withMainRenderTarget(framebuffer) {
            framebuffer.resizeLazy(viewport.width, viewport.height)

            framebuffer.bindWrite(true)

            val levelRenderer = Manager.getLevelRenderer(MC.level!!.dimension())

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
            
            afterRender.forEach { it(this) }

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