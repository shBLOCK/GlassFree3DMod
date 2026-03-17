package dev.shblock.glassfree3d

import dev.shblock.glassfree3d.blocks.PassthroughBlock
import dev.shblock.glassfree3d.cube_demo.CubeDemo1
import dev.shblock.glassfree3d.demo1.Demo1
import dev.shblock.glassfree3d.flipgo_demo.FlipGoDemo1
import dev.shblock.glassfree3d.utils.MC
import net.minecraft.core.Registry
import net.neoforged.bus.api.SubscribeEvent
import net.neoforged.fml.common.EventBusSubscriber
import net.neoforged.fml.common.Mod
import net.neoforged.neoforge.client.event.ClientTickEvent
import net.neoforged.neoforge.client.event.RenderFrameEvent
import net.neoforged.neoforge.registries.DeferredBlock
import net.neoforged.neoforge.registries.DeferredRegister
import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.Logger
import thedarkcolour.kotlinforforge.neoforge.forge.MOD_BUS
import thedarkcolour.kotlinforforge.neoforge.forge.getValue

@Mod(GlassFree3DMod.ID)
@EventBusSubscriber
object GlassFree3DMod {
    const val ID = "glassfree3d"

    val LOGGER: Logger = LogManager.getLogger(ID)

    init {
        Blocks.REGISTRY.register(MOD_BUS)
        Items.REGISTRY.register(MOD_BUS)
    }

    @SubscribeEvent
    fun onPreRenderFrame(event: RenderFrameEvent.Pre) {
        if (MC.level != null) {
            Demo1.tick()
//            CubeDemo1.tick()
//            FlipGoDemo1.tick()
        }
    }

    object Blocks {
        val REGISTRY = DeferredRegister.createBlocks(ID)
        val PASSTHROUGH_BLOCK = REGISTRY.register("passthrough", ::PassthroughBlock)
    }

    object Items {
        val REGISTRY = DeferredRegister.createItems(ID)
        val PASSTHROUGH_BLOCK = REGISTRY.registerSimpleBlockItem(Blocks.PASSTHROUGH_BLOCK)
    }
}
