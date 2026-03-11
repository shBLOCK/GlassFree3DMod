package dev.shblock.glassfree3d.utils

import com.google.gson.JsonObject
import com.mojang.blaze3d.pipeline.RenderTarget
import dev.shblock.glassfree3d.ducks.MinecraftAccessor
import net.minecraft.client.Minecraft
import net.minecraft.client.renderer.Rect2i
import net.minecraft.world.phys.Vec3
import org.joml.Vector2i
import org.joml.Vector3d
import org.joml.Vector3f
import org.lwjgl.glfw.GLFW

inline val MC: Minecraft get() = Minecraft.getInstance()
inline val MCA: MinecraftAccessor get() = Minecraft.getInstance() as MinecraftAccessor

object MiscUtils {
    fun withMainRenderTarget(renderTarget: RenderTarget, block: () -> Unit) {
        val org = MC.mainRenderTarget
        MCA.gf_setMainRenderTarget(renderTarget)
        block()
        MCA.gf_setMainRenderTarget(org)
    }

    fun getMonitorPos(monitor: Long): Vector2i {
        GLFW.glfwGetMonitors()
        val x = intArrayOf(1)
        val y = intArrayOf(1)
        GLFW.glfwGetMonitorPos(monitor, x, y)
        return Vector2i(x[0], y[0])
    }
}

inline fun Vec3.toVector3d() = Vector3d(x, y, z)
inline fun Vector3d.toVec3() = Vec3(x, y, z)
inline fun Vector3d.toVector3f() = Vector3f(x.toFloat(), y.toFloat(), z.toFloat())
inline operator fun Vector3d.plus(other: Vector3d): Vector3d = add(other, Vector3d())
inline operator fun Vector3d.minus(other: Vector3d): Vector3d = sub(other, Vector3d())
inline operator fun Vector3d.times(other: Vector3d): Vector3d = mul(other, Vector3d())
inline val Rect2i.x1 get() = x + width
inline val Rect2i.y1 get() = y + height

inline val Double.Companion.HALF_PI get() = Math.PI / 2.0
inline val Double.Companion.PI get() = Math.PI
inline val Double.Companion.TAU get() = Math.TAU

val JsonObject.asVector3d get() = Vector3d(get("x").asDouble, get("y").asDouble, get("z").asDouble)

fun RenderTarget.resizeLazy(width: Int, height: Int) {
    if (this.width != width || this.height != height) {
        this.resize(width, height, false)
    }
}