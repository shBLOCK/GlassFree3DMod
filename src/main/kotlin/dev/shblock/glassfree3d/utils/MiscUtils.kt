package dev.shblock.glassfree3d.utils

import com.mojang.blaze3d.pipeline.RenderTarget
import dev.shblock.glassfree3d.ducks.MinecraftAccessor
import net.minecraft.client.Minecraft

inline val MC: Minecraft get() = Minecraft.getInstance()
inline val MCA: MinecraftAccessor get() = Minecraft.getInstance() as MinecraftAccessor

object MiscUtils {
    fun withMainRenderTarget(renderTarget: RenderTarget, block: () -> Unit) {
        val org = MC.mainRenderTarget
        MCA.gf_setMainRenderTarget(renderTarget)
        block()
        MCA.gf_setMainRenderTarget(org)
    }
}
