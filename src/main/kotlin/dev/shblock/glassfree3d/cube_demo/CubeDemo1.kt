package dev.shblock.glassfree3d.cube_demo

import com.mojang.blaze3d.vertex.Tesselator
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
import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

class Visualizer(private val window: ModWindow) {
    private val tesselator = Tesselator(4096)

    fun draw() {
        window.makeCurrent()

        window.endFrame()
    }
}

object CubeDemo1 {
    private lateinit var visualizer: Visualizer
    val realCameraPos = Vector3d(0.0, 0.0, 1.0)
    val cubeRealPose = Screen3D.Pose()
    val cubeVirtualPose = Screen3D.Pose()
    val cubeScreens = mutableListOf<Screen3D>()

    private var eyePos = AtomicReference(Vector3d(10.0, 10.0, 20.0))

    private var initialized = false

    fun init() {
        initialized = true

        Screen3D.Manager.afterRenderAll += { visualizer.draw() }

        visualizer = Visualizer(ModWindow(Vector2i(800, 800), title = "Visualizer"))

        val RESOLUTION = 480
        val SIZE = 7.1
        val FACE_OFFSET = 4.2
        val cubeWindow = ModWindow(Vector2i(RESOLUTION * 5, RESOLUTION), title = "Cube Demo 1")
        fun makeCubeScreen(index: Int, orientation: Quaterniond): Screen3D {
            val realPose = Screen3D.Pose(
                orientation.transform(Vector3d(0.0, 0.0, FACE_OFFSET)),
                orientation,
                parent = cubeRealPose
            )
            return Screen3D(
                cubeWindow, Rect2i(RESOLUTION * index, 0, RESOLUTION, RESOLUTION),
                realPose = realPose,
                realSize = Vector2d(SIZE),
                virtualPose = realPose.copy(parent = cubeVirtualPose),
                virtualSize = Vector2d(1.0),
                realCameraPos = realCameraPos,
            )
        }
        cubeScreens += makeCubeScreen(0, Quaterniond()) // front
        cubeScreens += makeCubeScreen(1, Quaterniond().rotateX(-Double.HALF_PI)) // top
        cubeScreens += makeCubeScreen(2, Quaterniond().rotateY(Double.HALF_PI)) // right
        cubeScreens += makeCubeScreen(3, Quaterniond().rotateY(Double.HALF_PI * 2.0)) // back
        cubeScreens += makeCubeScreen(4, Quaterniond().rotateY(Double.HALF_PI * 3.0)) // left

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

        cubeScreens.forEach {
            it.clipAtScreenPlane = false
            it.zNear = 0.05
        }
        cubeVirtualPose.scale = 8.0

        realCameraPos.set(
            Vector3d(eyePos.get())
                .mul(Vector3d(1.0, 1.0, -1.0))
                .rotate(Quaterniond())
                .add(Vector3d(0.0, 0.0, -10.0))
        )
        println(realCameraPos)

        cubeVirtualPose.pos = MC.gameRenderer.mainCamera.position.toVector3d()
        cubeVirtualPose.orientation = Quaterniond(MC.gameRenderer.mainCamera.rotation())
    }
}