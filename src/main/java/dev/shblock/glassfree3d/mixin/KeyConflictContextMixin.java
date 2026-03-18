package dev.shblock.glassfree3d.mixin;

import net.minecraft.client.Minecraft;
import net.neoforged.neoforge.client.settings.KeyConflictContext;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(targets = "net.neoforged.neoforge.client.settings.KeyConflictContext$3") // IN_GAME
public class KeyConflictContextMixin {
    @Inject(method = "isActive", at = @At("HEAD"), cancellable = true)
    private void gf_doNotDisableInGameKeybindsIfMainWindowNotFocused(CallbackInfoReturnable<Boolean> cir) {
        //noinspection ConstantValue
        if ((Object) this != KeyConflictContext.IN_GAME) throw new AssertionError();
        
        if (!Minecraft.getInstance().isWindowActive())
            cir.setReturnValue(true);
    }
}
