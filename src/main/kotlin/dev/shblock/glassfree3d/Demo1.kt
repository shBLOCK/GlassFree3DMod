package dev.shblock.glassfree3d

import dev.shblock.glassfree3d.rendering.ModWindow
import dev.shblock.glassfree3d.rendering.Screen3D
import net.minecraft.client.renderer.Rect2i
import org.joml.Vector2i
import org.lwjgl.glfw.GLFW

object Demo1 {
    val window = ModWindow(Vector2i(1920, 1080), title = "Demo1")
    val screen = Screen3D(window, Rect2i(100, 100, 1280, 720))

    fun tick() {

    }
}