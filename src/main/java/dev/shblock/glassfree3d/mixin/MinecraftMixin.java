package dev.shblock.glassfree3d.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import dev.shblock.glassfree3d.ducks.MinecraftAccessor;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.Overlay;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.player.LocalPlayer;
import org.objectweb.asm.Opcodes;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Constant;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyConstant;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import javax.annotation.Nullable;

@Mixin(Minecraft.class)
public abstract class MinecraftMixin implements MinecraftAccessor {
    @Final
    @Shadow
    @Mutable
    private RenderTarget mainRenderTarget;

    @Override
    public void gf_setMainRenderTarget(RenderTarget renderTarget) {
        mainRenderTarget = renderTarget;
    }

    @Shadow @Nullable private Overlay overlay;

    @Shadow protected abstract void handleKeybinds();

    @Shadow protected int missTime;

    @Shadow public abstract boolean isWindowActive();

    @Shadow @Nullable public Screen screen;

    @Shadow @Nullable public ClientLevel level;

    @Shadow @Nullable public LocalPlayer player;
    
    @Inject(method = "tick", at = @At(value = "FIELD", target = "Lnet/minecraft/client/Minecraft;overlay:Lnet/minecraft/client/gui/screens/Overlay;", opcode = Opcodes.GETFIELD), allow = 1)
    private void gf_alwaysHandleKeyBindWhenMainWindowNotFocused(CallbackInfo ci) {
        if (this.isWindowActive()) return;
        if (this.level == null || this.player == null) return;
        if (this.overlay != null || this.screen != null) {
            this.handleKeybinds();
            if (this.missTime > 0) {
                --this.missTime;
            }
        }
    }

    @ModifyConstant(method = "tick", constant = @Constant(intValue = 10000), allow = 1)
    private int gf_dontResetMissTimeWhenMainWindowNotFocused(int constant) {
        if (this.isWindowActive()) return constant;
        return this.missTime;
    }
}
