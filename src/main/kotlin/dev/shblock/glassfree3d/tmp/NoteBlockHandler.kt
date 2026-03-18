package dev.shblock.glassfree3d.tmp

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import net.minecraft.client.Minecraft
import net.minecraft.world.level.block.state.properties.NoteBlockInstrument
import net.neoforged.bus.api.SubscribeEvent
import net.neoforged.fml.common.EventBusSubscriber
import net.neoforged.neoforge.event.level.NoteBlockEvent
import java.net.DatagramPacket
import java.net.DatagramSocket
import javax.sound.midi.MidiMessage
import javax.sound.midi.MidiSystem
import javax.sound.midi.ShortMessage

@EventBusSubscriber
object NoteBlockHandler {
    init {
        MidiSystem.getMidiDeviceInfo().forEach { println("${it.name}, ${it.vendor}, ${it.version}, ${it.description}") }
    }

    val device = MidiSystem.getMidiDevice(MidiSystem.getMidiDeviceInfo().find { it.name == "loopMIDI Port" }!!)
        .also { it.open() }
    val deviceReceiver = device.receiver
    
    @OptIn(DelicateCoroutinesApi::class, ExperimentalCoroutinesApi::class)
    val coroutineScope = CoroutineScope(newSingleThreadContext("NoteBlockHandler"))
    
    private fun playNote(note: Int, channel: Int, duration: Int) {
        coroutineScope.launch {
            deviceReceiver.send(ShortMessage(ShortMessage.NOTE_ON, channel, note, 100), -1)
            delay(duration.toLong())
            deviceReceiver.send(ShortMessage(ShortMessage.NOTE_OFF, channel, note, 0), -1)
        }
    }
    
    @SubscribeEvent
    fun onNoteBlockPlay(event: NoteBlockEvent.Play) {
        when (event.instrument) {
            NoteBlockInstrument.BIT -> playNote(54 + event.vanillaNoteId, 0, 100)
            NoteBlockInstrument.BANJO -> playNote(54 + event.vanillaNoteId, 1, 100)
            else -> return
        }
        event.isCanceled = true
    }
}