package dev.shblock.glassfree3d.rendering

import com.mojang.blaze3d.systems.RenderSystem
import com.mojang.blaze3d.vertex.Tesselator
import dev.shblock.glassfree3d.utils.MC
import org.joml.Vector2i
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.opengl.GL
import org.lwjgl.system.MemoryUtil.NULL

class ModWindow(
    size: Vector2i,
    pos: Vector2i? = null,
    title: String
) {
    private var _size = size
    var size
        get() = _size
        set(value) {
            TODO()
        }

    private var _pos = pos
    var pos
        get() = _pos!!
        set(value) {
            TODO()
        }

    init {
        glfwDefaultWindowHints()
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API)
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2)
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, 1)
        glfwWindowHint(GLFW_POSITION_X, pos?.x ?: GLFW_ANY_POSITION)
        glfwWindowHint(GLFW_POSITION_Y, pos?.y ?: GLFW_ANY_POSITION)
    }

    private val window = glfwCreateWindow(size.x, size.y, title, NULL, MC.window.window)

    init {
        val oldCtx = glfwGetCurrentContext()

        glfwMakeContextCurrent(window)
        GL.createCapabilities()
        val maxTex = RenderSystem.maxSupportedTextureSize()
        glfwSetWindowSizeLimits(window, -1, -1, maxTex, maxTex)

        if (_pos == null) {
            val x = intArrayOf(0)
            val y = intArrayOf(0)
            glfwGetWindowPos(window, x, y)
            _pos = Vector2i(x[0], y[0])
        }

        glfwMakeContextCurrent(oldCtx)
    }

    fun makeCurrent() {
        glfwMakeContextCurrent(window)
    }

    fun endFrame() {
        RenderSystem.replayQueue()
        Tesselator.getInstance().clear()
        glfwSwapBuffers(window)
    }
}