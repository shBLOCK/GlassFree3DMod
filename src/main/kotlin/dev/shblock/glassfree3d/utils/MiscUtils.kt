package dev.shblock.glassfree3d.utils

import com.mojang.blaze3d.pipeline.RenderTarget
import dev.shblock.glassfree3d.ducks.MinecraftAccessor
import net.minecraft.client.Minecraft
import net.minecraft.world.phys.Vec3
import org.joml.Vector2i
import org.joml.Vector3d
import org.joml.Vector3dc
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

fun Vec3.toVector3d() = Vector3d(x, y, z)
fun Vector3dc.toVec3() = Vec3(x(), y(), z())
operator fun Vector3dc.plus(other: Vector3dc): Vector3d = add(other, Vector3d())
operator fun Vector3dc.minus(other: Vector3dc): Vector3d = sub(other, Vector3d())
operator fun Vector3dc.times(other: Vector3dc): Vector3d = mul(other, Vector3d())
