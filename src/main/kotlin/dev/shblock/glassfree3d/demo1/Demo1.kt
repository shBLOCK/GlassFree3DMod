package dev.shblock.glassfree3d.demo1

import com.mojang.blaze3d.platform.InputConstants
import dev.shblock.glassfree3d.ducks.GameRendererAccessor
import dev.shblock.glassfree3d.rendering.ModWindow
import dev.shblock.glassfree3d.rendering.Screen3D
import dev.shblock.glassfree3d.utils.MC
import dev.shblock.glassfree3d.utils.MCA
import dev.shblock.glassfree3d.utils.asVector3d
import dev.shblock.glassfree3d.utils.plus
import dev.shblock.glassfree3d.utils.toVec3
import dev.shblock.glassfree3d.utils.toVector3d
import net.minecraft.client.KeyMapping
import net.minecraft.client.renderer.Rect2i
import net.minecraft.util.GsonHelper
import net.minecraft.world.entity.projectile.ProjectileUtil
import net.minecraft.world.level.ClipContext
import net.minecraft.world.phys.AABB
import net.minecraft.world.phys.HitResult
import org.joml.Quaterniond
import org.joml.Vector2d
import org.joml.Vector2i
import org.joml.Vector3d
import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

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

        window = ModWindow(Vector2i(2560, 1600), title = "Demo1")
        window.mouseButtonCallback = { _, button, action, _ ->
            MC.execute {
                val isPress = action == InputConstants.PRESS
                val key = InputConstants.Type.MOUSE.getOrCreate(button)
                KeyMapping.set(key, isPress)
                if (isPress) {
                    KeyMapping.click(key)
                }
            }
        }
        window.scrollCallback = { _, offset ->
            MC.execute {
                if (MC.overlay == null && MC.screen == null) {
                    MC.mouseHandler.onScroll(MC.window.window, offset.x, offset.y)
                }
            }
        }

        (MC.gameRenderer as GameRendererAccessor).gf_addPicker {
            if (!window.focused) return@gf_addPicker null

            val origin = screen.virtualCameraPos
            val direction = screen.unprojectVirtualScreen(window.toNDC(window.cursorPos))
            var maxDistance = 1e3
            val localFrom = Vector3d(direction).mul(screen.zNear)
            var localTo = Vector3d(direction).mul(maxDistance).add(localFrom)
            val globalFrom = Vector3d(localFrom).add(origin)
            var globalTo = Vector3d(localTo).add(origin)

            val blockHitResult =
                MC.level!!.clip(
                    ClipContext(
                        globalFrom.toVec3(), globalTo.toVec3(),
                        ClipContext.Block.OUTLINE,
                        ClipContext.Fluid.NONE,
                        MC.player!!
                    )
                )

            val blockDistance = blockHitResult.location.distanceTo(globalFrom.toVec3())
            if (blockHitResult.type != HitResult.Type.MISS) {
                maxDistance = blockDistance
                localTo = Vector3d(direction).mul(maxDistance).add(localFrom)
                globalTo = Vector3d(localTo).add(origin)
            }
            val entityHitResult = ProjectileUtil.getEntityHitResult(
                MC.player!!,
                globalFrom.toVec3(), globalTo.toVec3(),
                AABB(globalFrom.toVec3(), globalTo.toVec3()).inflate(1.0),
                { !it.isSpectator && it.isPickable },
                maxDistance * maxDistance
            )
            if (entityHitResult != null && entityHitResult.location.distanceTo(globalFrom.toVec3()) < blockDistance) {
                entityHitResult
            } else blockHitResult
        }

        screen = Screen3D(window, Rect2i(0, 0, 2560, 1600))

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
//                        eyePos.set(rightEye)
                    }

//                    println("Connecting...")
//                    Socket("127.0.0.1", 30001).use { socket ->
//                        println("Connected to ${socket.remoteSocketAddress}")
//                        val reader = BufferedReader(InputStreamReader(socket.inputStream))
//                        while (true) {
//                            val pkt = GsonHelper.parse(reader.readLine())
//                            val leftEye = pkt.get("left_eye_3d").asJsonObject.asVector3d
//                            val rightEye = pkt.get("right_eye_3d").asJsonObject.asVector3d
//                            eyePos.set((leftEye + rightEye) / 2.0)
//                        }
//                    }
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
        screen.realPose.pos = Vector3d(
            screenA.add(screenB, Vector2d()).div(2.0).mul(1.0, -1.0).mul(DISPLAY_SIZE),
            0.0
        ).add(DISPLAY_POS)
        screen.realSize = screenB.sub(screenA, Vector2d()).mul(DISPLAY_SIZE)
        screen.realCameraPos = Vector3d(eyePos.get())
            .mul(Vector3d(1.0, 1.0, -1.0))
            .rotate(CAMERA_ORIENTATION)
            .add(CAMERA_POS)
//        println(screen.realCameraPos)

        screen.virtualPose.pos = MC.gameRenderer.mainCamera.position.toVector3d()
        screen.virtualPose.orientation = Quaterniond(MC.gameRenderer.mainCamera.rotation())
        screen.virtualSize = screen.realSize.mul(16.0 / DISPLAY_SIZE.y, Vector2d())
    }
}