package dev.shblock.glassfree3d

import dev.shblock.glassfree3d.utils.MC
import net.neoforged.bus.api.SubscribeEvent
import net.neoforged.fml.common.EventBusSubscriber
import net.neoforged.fml.common.Mod
import net.neoforged.neoforge.client.event.ClientTickEvent
import org.apache.logging.log4j.LogManager
import org.apache.logging.log4j.Logger

@Mod(GlassFree3DMod.ID)
@EventBusSubscriber
object GlassFree3DMod {
    const val ID = "glassfree3d"

    val LOGGER: Logger = LogManager.getLogger(ID)

    @SubscribeEvent
    fun onClientTick(event: ClientTickEvent.Pre) {
        if (MC.level != null) {
            Demo1.tick()
        }
    }
}
