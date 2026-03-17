package dev.shblock.glassfree3d.mixin;

import dev.shblock.glassfree3d.ducks.GameRendererAccessor;
import dev.shblock.glassfree3d.rendering.Screen3D;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.world.phys.EntityHitResult;
import net.minecraft.world.phys.HitResult;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Objects;
import java.util.function.Supplier;

@Mixin(GameRenderer.class)
public class GameRendererMixin implements GameRendererAccessor {
    @Shadow
    @Final
    private Minecraft minecraft;

    @Inject(
        method = "render",
        at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/pipeline/RenderTarget;bindWrite(Z)V")
    )
    void onAfterNormalLevelRender(DeltaTracker deltaTracker, boolean renderLevel, CallbackInfo ci) {
        Screen3D.Manager.INSTANCE.renderAll$glassfree3d();
    }

    @Unique
    public Collection<Supplier<HitResult>> pickers = new ArrayList<>();

    @Inject(method = "pick(F)V", at = @At("TAIL"), order = 1100)
    private void overridePick(float partialTicks, CallbackInfo ci) {
        pickers.stream().map(Supplier::get).filter(Objects::nonNull).findFirst().ifPresent(hitResult -> {
            minecraft.hitResult = hitResult;
            if (hitResult instanceof EntityHitResult entityHitResult) {
                minecraft.crosshairPickEntity = entityHitResult.getEntity();
            }
        });
    }

    @Override
    public void gf_addPicker(Supplier<HitResult> picker) {
        pickers.add(picker);
    }
}
