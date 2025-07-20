package dev.shblock.glassfree3d

import com.google.gson.JsonObject
import dev.shblock.glassfree3d.rendering.ModWindow
import dev.shblock.glassfree3d.rendering.Screen3D
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.plus
import dev.shblock.glassfree3d.utils.toVector3d
import net.minecraft.client.renderer.Rect2i
import net.minecraft.util.GsonHelper
import org.joml.Vector2d
import org.joml.Vector2i
import org.joml.Vector3d
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.net.Socket
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

private val JsonObject.asVector3d get() = Vector3d(get("x").asDouble, get("y").asDouble, get("z").asDouble)

private val DISPLAY_RESOLUTION = Vector2i(2560, 1600)
private val DISPLAY_SIZE = Vector2d(34.5, 21.5)
private val DISPLAY_CENTER_OFFSET_FROM_CAMERA = Vector3d(0.0, -DISPLAY_SIZE.y / 2 - 0.6, 0.0)

object Demo1 {
    private lateinit var window: ModWindow
    private lateinit var screen: Screen3D

    private var initialized = false

    private var eyePos = AtomicReference(Vector3d(0.0, 0.0, 1.0))

    fun init() {
        initialized = true

        window = ModWindow(Vector2i(1920, 1200), title = "Demo1")
        screen = Screen3D(window, Rect2i(0, 0, 1920, 1200))

        thread(name = "Demo1SocketClient") {
            while (true) {
                try {
                    println("Connecting...")
                    Socket("127.0.0.1", 30001).use { socket ->
                        println("Connected to ${socket.remoteSocketAddress}")
                        val reader = BufferedReader(InputStreamReader(socket.inputStream))
                        while (true) {
                            val pkt = GsonHelper.parse(reader.readLine())
                            val leftEye = pkt.get("left_eye_3d").asJsonObject.asVector3d
                            val rightEye = pkt.get("right_eye_3d").asJsonObject.asVector3d
                            eyePos.set((leftEye + rightEye) / 2.0)
                        }
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

        screen.viewport = Rect2i(0, 0, window.framebufferSize.x, window.framebufferSize.y)
        val res = Vector2d(DISPLAY_RESOLUTION)
        val screenA = (Vector2d(window.pos) / res).sub(0.5, 0.5)
        val screenB = (Vector2d(window.pos.add(window.size, Vector2i())) / res).sub(0.5, 0.5)
        screen.realPos = DISPLAY_CENTER_OFFSET_FROM_CAMERA + Vector3d(
            screenA.add(screenB, Vector2d()).div(2.0).mul(1.0, -1.0).mul(DISPLAY_SIZE),
            0.0
        )
        screen.realSize = screenB.sub(screenA, Vector2d()).mul(DISPLAY_SIZE)
        screen.realCameraPos = Vector3d(eyePos.get()).mul(Vector3d(-1.0, 1.0, -1.0))

        screen.virtualPos = MC.player!!.eyePosition.toVector3d()
        screen.virtualSize = screen.realSize.mul(1.0 / DISPLAY_SIZE.y, Vector2d())
    }
}