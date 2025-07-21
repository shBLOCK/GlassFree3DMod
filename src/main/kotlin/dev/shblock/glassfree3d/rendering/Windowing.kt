package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.Tesselator
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.MiscUtils
import org.joml.Vector2i
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.opengl.GL
import org.lwjgl.system.MemoryUtil.NULL

class ModWindow(
    size: Vector2i,
    pos: Vector2i = Vector2i(GLFW_ANY_POSITION),
    val title: String
) {
    private var _size = size
    var size
        get() = _size
        set(value) {
            TODO()
        }

    private var _pos = pos
    var pos
        get() = _pos
        set(value) {
            TODO()
        }

    var framebufferSize = Vector2i(1, 1)
        private set

    init {
        glfwDefaultWindowHints()
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API)
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2)
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 1)
        glfwWindowHint(GLFW_POSITION_X, pos.x)
        glfwWindowHint(GLFW_POSITION_Y, pos.y)
    }

    private val window = glfwCreateWindow(size.x, size.y, title, NULL, MC.window.window)

    init {
        glfwSetWindowPosCallback(window) { _, x, y ->
            _pos = Vector2i(x, y)
        }
        glfwSetWindowSizeCallback(window) { _, width, height ->
            _size = Vector2i(width, height)
        }
        glfwSetFramebufferSizeCallback(window) { _, width, height ->
            framebufferSize = Vector2i(width, height)
        }

        val oldCtx = glfwGetCurrentContext()

        glfwMakeContextCurrent(window)
        GL.createCapabilities()
        val maxTex = RenderSystem.maxSupportedTextureSize()
        glfwSetWindowSizeLimits(window, -1, -1, maxTex, maxTex)

        val tmpX = intArrayOf(0)
        val tmpY = intArrayOf(0)
        glfwGetWindowSize(window, tmpX, tmpY)
        _size = Vector2i(tmpX[0], tmpY[0])
        glfwGetWindowPos(window, tmpX, tmpY)
        _pos = Vector2i(tmpX[0], tmpY[0])
        glfwGetFramebufferSize(window, tmpX, tmpY)
        framebufferSize = Vector2i(tmpX[0], tmpY[0])

        glfwMakeContextCurrent(oldCtx)
    }

    fun posInMonitor(monitor: Long): Vector2i = Vector2i(pos).sub(MiscUtils.getMonitorPos(monitor))

    fun makeCurrent() {
        glfwMakeContextCurrent(window)
    }

    fun endFrame() {
        RenderSystem.replayQueue()
        Tesselator.getInstance().clear()
        glfwSwapBuffers(window)
    }
}