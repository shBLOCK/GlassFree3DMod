package dev.shblock.glassfree3d.flipgo_demo

import dev.shblock.glassfree3d.rendering.ModWindow
import dev.shblock.glassfree3d.rendering.Screen3D
import dev.shblock.glassfree3d.utils.HALF_PI
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.asVector3d
import dev.shblock.glassfree3d.utils.plus
import dev.shblock.glassfree3d.utils.toVector3d
import net.minecraft.client.renderer.Rect2i
import net.minecraft.util.GsonHelper
import org.joml.Quaterniond
import org.joml.Vector2d
import org.joml.Vector2i
import org.joml.Vector3d
import org.lwjgl.glfw.GLFW
import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

object FlipGoDemo1 {
    val screens = mutableListOf<Screen3D>()

    private var eyePos = AtomicReference(Vector3d(10.0, 10.0, 20.0))

    val realCameraPos = Vector3d(0.0, 0.0, 0.0)
    val virtualPose = Screen3D.Pose()

    var initialized = false

    fun init() {
        initialized = true
        val RESOLUTION = Vector2i(2560, 1600)
        val REAL_SIZE = Vector2d(34.5, 21.5)
        val SCREEN_EDGE = 0.6
        val VIRTUAL_SIZE = Vector2d(REAL_SIZE.x / REAL_SIZE.y, 1.0)
        val VIRTUAL_SCALE = VIRTUAL_SIZE.y / REAL_SIZE.y
        val displays = GLFW.glfwGetMonitors()!!
        run {
            screens += Screen3D(
                ModWindow(RESOLUTION, title = "Upper", fullScreenMonitor = displays[2]),
                Rect2i(0, 0, RESOLUTION.x, RESOLUTION.y),
                realSize = REAL_SIZE,
                virtualSize = VIRTUAL_SIZE,
                virtualPose = Screen3D.Pose(parent = virtualPose),
                realCameraPos = realCameraPos
            )
        }
        run {
            val pose = Screen3D.Pose(
                pos = Vector3d(0.0, -(REAL_SIZE.y / 2 + SCREEN_EDGE), REAL_SIZE.y / 2 + SCREEN_EDGE),
                orientation = Quaterniond().rotateX(-Double.HALF_PI)
            )
            screens += Screen3D(
                ModWindow(RESOLUTION, title = "Lower", fullScreenMonitor = displays[1]),
                Rect2i(0, 0, RESOLUTION.x, RESOLUTION.y),
                realSize = REAL_SIZE,
                virtualSize = VIRTUAL_SIZE,
                realPose = pose,
                virtualPose = pose.copy().apply {
                    parent = virtualPose
                    pos.mul(VIRTUAL_SCALE)
                },
                realCameraPos = realCameraPos
            )
        }

        thread(name = "Demo1SocketClient", isDaemon = true) {
            while (true) {
                try {
                    val socket = DatagramSocket(30001)
                    val buffer = ByteArray(1024)

                    while (true) {
                        val pkt = DatagramPacket(buffer, buffer.size)
                        socket.receive(pkt)
                        val string = String(pkt.data, 0, pkt.length)
                        val data = GsonHelper.parse(string)
                        val leftEye = data.get("left_eye_3d").asJsonObject.asVector3d
                        val rightEye = data.get("right_eye_3d").asJsonObject.asVector3d
                        eyePos.set((leftEye + rightEye) / 2.0)
                    }
                } catch (e: IOException) {
                    println("Failed to receive packet: $e")
                    Thread.sleep(1000)
                }
            }
        }
    }

    fun tick() {
        if (!initialized) init()

        screens.forEach {
            it.zNear = 0.05
            it.clipAtScreenPlane = false
        }
        virtualPose.scale = 8.0

        realCameraPos.set(
            Vector3d(eyePos.get())
                .mul(Vector3d(1.0, 1.0, -1.0))
                .add(0.0, 12.75, 2.0)
        )

        virtualPose.pos = MC.gameRenderer.mainCamera.position.toVector3d()
        virtualPose.orientation = Quaterniond(MC.gameRenderer.mainCamera.rotation())
    }
}