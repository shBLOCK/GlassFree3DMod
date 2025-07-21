package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.pipeline.MainTarget
import com.mojang.blaze3d.platform.GlStateManager
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.*
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
import org.joml.Matrix4d
import org.joml.Matrix4f
import org.joml.Quaterniond
import org.joml.Quaternionf
import org.joml.Vector2d
import org.joml.Vector3d
import org.lwjgl.glfw.GLFW.glfwMakeContextCurrent
import java.lang.Runtime
import java.util.*

class Screen3D(
    val window: ModWindow,
    var viewport: Rect2i
) {
    private val framebuffer = MainTarget(viewport.width, viewport.height)

    var virtualPos = Vector3d()
    var virtualSize = Vector2d(1.0)
    var virtualOrientation = Quaterniond()
    var realPos = Vector3d()
    var realSize = Vector2d(1.0)
    var realOrientation = Quaterniond()
    var realCameraPos = Vector3d(0.0, 0.0, 1.0)
    var virtualCameraPos = Vector3d(0.0, 0.0, 1.0)
        private set
    var zNear = 0.05
    var clipAtScreenPlane = true

    private val virtualCamera = Camera()
    private var frustumMatrix = Matrix4d()
    private var projectionMatrix = Matrix4d()

    init {
        Manager.newScreen(this)
    }

    private fun updateProjectionAndCamera() {
        val localRealCameraPos = realOrientation.transformInverse(realCameraPos - realPos)
        val scale = virtualSize.div(realSize, Vector2d())
        val scale3d = Vector3d(scale, (scale.x + scale.y) / 2.0)
        val localVirtualCameraPos = localRealCameraPos.mul(scale3d, Vector3d())
        virtualCameraPos = virtualOrientation.transform(localVirtualCameraPos, Vector3d()) + virtualPos

        if (clipAtScreenPlane) {
            zNear = localVirtualCameraPos.z
        }
        virtualCamera.initialized = true
        virtualCamera.position = virtualCameraPos.toVec3()
        virtualCamera.rotation.set(virtualOrientation)
        frustumMatrix.rotation(Quaternionf(virtualOrientation.conjugate(Quaterniond())))
        val halfVirtualSize = virtualSize.div(2.0, Vector2d())
        val left = (-localVirtualCameraPos.x - halfVirtualSize.x) / localVirtualCameraPos.z
        val right = (-localVirtualCameraPos.x + halfVirtualSize.x) / localVirtualCameraPos.z
        val bottom = (-localVirtualCameraPos.y - halfVirtualSize.y) / localVirtualCameraPos.z
        val top = (-localVirtualCameraPos.y + halfVirtualSize.y) / localVirtualCameraPos.z
        projectionMatrix.setFrustum(
            left, right, bottom, top,
            zNear, MC.gameRenderer.depthFar.toDouble()
        )
    }

    private fun render() {
        RenderSystem.assertOnRenderThread()

        MiscUtils.withMainRenderTarget(framebuffer) {
            if (framebuffer.width != viewport.width || framebuffer.height != viewport.height) {
                framebuffer.resize(viewport.width, viewport.height, false)
            }

            framebuffer.bindWrite(true)

            val levelRenderer = Manager.getLevelRenderer(MC.level!!.dimension())
            virtualCamera.level = levelRenderer.level!!
            virtualCamera.entity = MC.player!!

            updateProjectionAndCamera()
            val frustumMatrixF = Matrix4f(frustumMatrix)
            val projectionMatrixF = Matrix4f(projectionMatrix)

            RenderSystem.setProjectionMatrix(projectionMatrixF, VertexSorting.DISTANCE_TO_ORIGIN)

            levelRenderer.prepareCullFrustum(
                virtualCamera.position,
                frustumMatrixF,
                projectionMatrixF
            )
            levelRenderer.renderLevel(
                MC.timer,
                false,
                virtualCamera,
                MC.gameRenderer,
                MC.gameRenderer.lightTexture(),
                frustumMatrixF,
                projectionMatrixF
            )

            framebuffer.unbindWrite()
        }
    }

    // Vertex buffer is not shared between contexts, the vertex format contains a vertex buffer, so we create our own here.
    private val SCREEN_BLIT_VF = VertexFormat.builder().add("Position", VertexFormatElement.POSITION).build()

    private fun blit() {
        // MainTarget.blitToScreen()
        RenderSystem.assertOnRenderThread()
        framebuffer.unbindWrite()
        GlStateManager._viewport(viewport.x, viewport.y, viewport.width, viewport.height)
        GlStateManager._colorMask(true, true, true, false)
        GlStateManager._disableDepthTest()
        GlStateManager._depthMask(false)
        GlStateManager._disableBlend()

        val minecraft = Minecraft.getInstance()
        val shaderinstance = Objects.requireNonNull<ShaderInstance?>(
            minecraft.gameRenderer.blitShader,
            "Blit shader not loaded"
        ) as ShaderInstance
        shaderinstance.setSampler("DiffuseSampler", framebuffer.colorTextureId)
        shaderinstance.apply()
        val bufferbuilder =
            RenderSystem.renderThreadTesselator().begin(VertexFormat.Mode.QUADS, SCREEN_BLIT_VF)
        bufferbuilder.addVertex(0.0f, 0.0f, 0.0f)
        bufferbuilder.addVertex(1.0f, 0.0f, 0.0f)
        bufferbuilder.addVertex(1.0f, 1.0f, 0.0f)
        bufferbuilder.addVertex(0.0f, 1.0f, 0.0f)
        BufferUploader.draw(bufferbuilder.buildOrThrow())
        shaderinstance.clear()
        GlStateManager._depthMask(true)
        GlStateManager._colorMask(true, true, true, true)
    }

    @Suppress("FunctionName")
    object Manager {
        private val screens = mutableListOf<Screen3D>()
        private val windows = mutableSetOf<ModWindow>()
        private val levelRenderers = mutableMapOf<ResourceKey<Level>, LevelRenderer>()

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
                ).apply { setLevel(MC.level) } // TODO: actually handle non-current levels
            }
        }

        internal fun renderAll() {
            screens.forEach { it.render() }

            RenderSystem.replayQueue()
            Tesselator.getInstance().clear()

            screens.groupBy(Screen3D::window).forEach { (window, windowScreens) ->
                window.makeCurrent()
                windowScreens.forEach { it.blit() }
                window.endFrame()
            }
            glfwMakeContextCurrent(MC.window.window)
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