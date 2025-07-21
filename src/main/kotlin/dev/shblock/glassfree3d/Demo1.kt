package dev.shblock.glassfree3d

import com.fazecast.jSerialComm.SerialPort
import com.google.gson.JsonObject
import dev.shblock.glassfree3d.rendering.ModWindow
import dev.shblock.glassfree3d.rendering.Screen3D
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.plus
import dev.shblock.glassfree3d.utils.toVector3d
import net.minecraft.client.renderer.Rect2i
import net.minecraft.util.GsonHelper
import org.joml.Quaterniond
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
private fun ByteArray.getFixedPoint16(offset: Int) =
    ((get(offset).toInt() or (get(offset + 1).toInt() shl 8)) - 32768).toDouble() / 32768.0
//    (get(offset).toInt() or (get(offset + 1).toInt() shl 8)).toDouble() / 65536.0

object Demo1 {
    private lateinit var window: ModWindow
    private lateinit var screen: Screen3D

    private var initialized = false

    private var eyePos = AtomicReference(Vector3d(0.0, 0.0, 1.0))
    private var sensorQuat = AtomicReference(Quaterniond())

    fun init() {
        initialized = true

        window = ModWindow(Vector2i(1920, 1200), title = "Demo1")
        screen = Screen3D(window, Rect2i(0, 0, 1920, 1200))

        thread(name = "Demo1SocketClient", isDaemon = true) {
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

//        thread(name = "Demo1WitmotionSensorThread", isDaemon = true) {
//            println("Serial thread")
//            val port = SerialPort.getCommPort("COM14")
//            port.setComPortParameters(921600, 8, 1, 0)
//
//            while (true) {
//                try {
//                    if (!port.openPort()) throw IOException("Failed to open serial port")
//                    println("Serial port opened")
//                    val buffer = ByteArray(9)
//                    while (true) {
//                        port.readBytes(buffer, 1)
//                        if (buffer[0] == 0x55.toByte()) {
//                            port.readBytes(buffer, 1)
//                            if (buffer[0] == 0x59.toByte()) {
//                                port.readBytes(buffer, 9)
//                                var sum = 0x55 + 0x59
//                                for (i in 0..7) {
//                                    sum += buffer[i]
//                                }
//                                println(sum and 0xFF)
//                                if ((sum and 0xFF) != buffer[8].toInt()) continue
////                                if (0x55.toByte() + 0x59.toByte() + buffer[0] + buffer[1] + buffer[0])
//                                sensorQuat.set(Quaterniond(
//                                    buffer.getFixedPoint16(0),
//                                    buffer.getFixedPoint16(2),
//                                    buffer.getFixedPoint16(4),
//                                    buffer.getFixedPoint16(6)
//                                ))
//                            }
//                        }
//                    }
//                } catch (e: IOException) {
//                    port.closePort()
//                    println("Serial port error: $e")
//                    e.printStackTrace()
//                }
//            }
//        }
    }

    fun tick() {
        if (!initialized) init()

        screen.clipAtScreenPlane = false
        screen.zNear = 1.0

        screen.viewport = Rect2i(0, 0, window.framebufferSize.x, window.framebufferSize.y)
        val res = Vector2d(DISPLAY_RESOLUTION)

//        val quat = Quaterniond(sensorQuat.get())
//        screen.realOrientation = quat
//        println(quat.lengthSquared())

        val windowPos = window.posInMonitor(DISPLAY_MONITOR)
        val screenA = (Vector2d(windowPos) / res).sub(0.5, 0.5)
        val screenB = (Vector2d(windowPos.add(window.size, Vector2i())) / res).sub(0.5, 0.5)
        screen.realPos = Vector3d(
            screenA.add(screenB, Vector2d()).div(2.0).mul(1.0, -1.0).mul(DISPLAY_SIZE),
            0.0
        ).add(DISPLAY_POS)
        screen.realSize = screenB.sub(screenA, Vector2d()).mul(DISPLAY_SIZE)
        screen.realCameraPos = Vector3d(eyePos.get())
            .mul(Vector3d(-1.0, 1.0, -1.0))
            .rotate(CAMERA_ORIENTATION)
            .add(CAMERA_POS)

        screen.virtualPos = MC.gameRenderer.mainCamera.position.toVector3d()
        screen.virtualOrientation = Quaterniond(MC.gameRenderer.mainCamera.rotation())
        screen.virtualSize = screen.realSize.mul(8.0 / DISPLAY_SIZE.y, Vector2d())
    }
}