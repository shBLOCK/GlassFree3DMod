package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.pipeline.MainTarget
import com.mojang.blaze3d.platform.GlStateManager
import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.*
import com.mojang.blaze3d.vertex.VertexFormat
import com.mojang.blaze3d.vertex.VertexFormatElement
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.MiscUtils
import net.minecraft.client.Minecraft
import net.minecraft.client.renderer.LevelRenderer
import net.minecraft.client.renderer.Rect2i
import net.minecraft.client.renderer.RenderBuffers
import net.minecraft.client.renderer.ShaderInstance
import net.minecraft.resources.ResourceKey
import net.minecraft.world.level.Level
import org.joml.Matrix4f
import org.joml.Quaternionf
import org.lwjgl.glfw.GLFW.glfwGetCurrentContext
import org.lwjgl.glfw.GLFW.glfwMakeContextCurrent
import java.util.*
import kotlin.math.PI

class Screen3D(
    val window: ModWindow,
    val viewport: Rect2i
) {
    private val framebuffer = MainTarget(viewport.width, viewport.height)
//    private val camera = Camera()

    init {
        Manager.newScreen(this)
    }

    private fun render() {
        RenderSystem.assertOnRenderThread()

        MiscUtils.withMainRenderTarget(framebuffer) {
            framebuffer.bindWrite(true)

            val levelRenderer = Manager.levelRenderers[MC.level!!.dimension()]!!

            val camera = MC.gameRenderer.mainCamera
            val frustumMatrix = Matrix4f().rotation(camera.rotation().conjugate(Quaternionf()))
            val projectionMatrix = Matrix4f()
            projectionMatrix.perspective(
                (PI / 2.0).toFloat(),
                (viewport.width / viewport.height).toFloat(),
                0.05F,
                MC.gameRenderer.depthFar
            )
            RenderSystem.setProjectionMatrix(projectionMatrix, VertexSorting.DISTANCE_TO_ORIGIN)

            levelRenderer.prepareCullFrustum(
                camera.position,
                frustumMatrix,
                projectionMatrix
            )
            levelRenderer.renderLevel(
                MC.timer,
                false,
                camera,
                MC.gameRenderer,
                MC.gameRenderer.lightTexture(),
                frustumMatrix,
                projectionMatrix
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

    object Manager {
        private val screens = mutableListOf<Screen3D>()
        private val windows = mutableSetOf<ModWindow>()
        internal val levelRenderers = mutableMapOf<ResourceKey<Level>, LevelRenderer>()

        init {
            levelRenderers[Level.OVERWORLD] = LevelRenderer(
                MC,
                MC.entityRenderDispatcher,
                MC.blockEntityRenderDispatcher,
                RenderBuffers(Runtime.getRuntime().availableProcessors())
            ).apply { setLevel(MC.level) }
        }

        internal fun newScreen(screen: Screen3D) {
            screens += screen
            windows += screen.window
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
    }
}