package dev.shblock.glassfree3d.mixin;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.Tesselator;
import dev.shblock.glassfree3d.rendering.Screen3D;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(GameRenderer.class)
public class GameRendererMixin {
    @Inject(
        method = "render",
        at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/pipeline/RenderTarget;bindWrite(Z)V")
    )
    void onAfterNormalLevelRender(DeltaTracker deltaTracker, boolean renderLevel, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.renderAll$glassfree3d();
    }
}
