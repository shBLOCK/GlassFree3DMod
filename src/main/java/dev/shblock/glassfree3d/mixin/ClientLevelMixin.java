package dev.shblock.glassfree3d.mixin;

import dev.shblock.glassfree3d.rendering.Screen3D;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import javax.annotation.Nullable;

@Mixin(ClientLevel.class)
public abstract class ClientLevelMixin {
    @Inject(
        method = "onChunkLoaded",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;onChunkLoaded(Lnet/minecraft/world/level/ChunkPos;)V", shift = At.Shift.AFTER)
    )
    void onChunkLoaded(ChunkPos chunkPos, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_onChunkLoaded$glassfree3d(((ClientLevel) (Object) this).dimension(), chunkPos);
    }

    @Inject(
        method = "sendBlockUpdated",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;blockChanged(Lnet/minecraft/world/level/BlockGetter;Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;Lnet/minecraft/world/level/block/state/BlockState;I)V", shift = At.Shift.AFTER)
    )
    void sendBlockUpdated(BlockPos pos, BlockState oldState, BlockState newState, int flags, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_blockChanged$glassfree3d((ClientLevel) (Object) this, pos, oldState, newState, flags);
    }

    @Inject(
        method = "setBlocksDirty",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;setBlockDirty(Lnet/minecraft/core/BlockPos;Lnet/minecraft/world/level/block/state/BlockState;Lnet/minecraft/world/level/block/state/BlockState;)V", shift = At.Shift.AFTER)
    )
    void setBlockDirty(BlockPos blockPos, BlockState oldState, BlockState newState, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_setBlockDirty$glassfree3d(((ClientLevel) (Object) this).dimension(), blockPos, oldState, newState);
    }

    @Inject(
        method = "setSectionDirtyWithNeighbors",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;setSectionDirtyWithNeighbors(III)V", shift = At.Shift.AFTER)
    )
    void setSectionDirtyWithNeighbors(int sectionX, int sectionY, int sectionZ, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_setSectionDirtyWithNeighbors$glassfree3d(((ClientLevel) (Object) this).dimension(), sectionX, sectionY, sectionZ);
    }

    @Inject(
        method = "destroyBlockProgress",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;destroyBlockProgress(ILnet/minecraft/core/BlockPos;I)V", shift = At.Shift.AFTER)
    )
    void destroyBlockProgress(int breakerId, BlockPos pos, int progress, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_destroyBlockProgress$glassfree3d(((ClientLevel) (Object) this).dimension(), breakerId, pos, progress);
    }

    @Inject(
        method = "globalLevelEvent",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;globalLevelEvent(ILnet/minecraft/core/BlockPos;I)V", shift = At.Shift.AFTER)
    )
    void globalLevelEvent(int id, BlockPos pos, int data, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_globalLevelEvent$glassfree3d(((ClientLevel) (Object) this).dimension(), id, pos, data);
    }

    @Inject(
        method = "levelEvent",
        at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/LevelRenderer;levelEvent(ILnet/minecraft/core/BlockPos;I)V", shift = At.Shift.AFTER)
    )
    void levelEvent(@Nullable Player player, int type, BlockPos pos, int data, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.LR_levelEvent$glassfree3d(((ClientLevel) (Object) this).dimension(), type, pos, data);
    }

//    @Inject(
//        method = "addParticle(Lnet/minecraft/core/particles/ParticleOptions;DDDDDD)V"
//    )
//
//    @Inject(
//        method = "addParticle(Lnet/minecraft/core/particles/ParticleOptions;ZDDDDDD)V"
//    )
//
//    @Inject(
//        method = "addAlwaysVisibleParticle(Lnet/minecraft/core/particles/ParticleOptions;DDDDDD)V"
//    )
//
//    @Inject(
//        method = "addAlwaysVisibleParticle(Lnet/minecraft/core/particles/ParticleOptions;ZDDDDDD)V"
//    )
}
