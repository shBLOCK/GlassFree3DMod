package dev.shblock.glassfree3d.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import dev.shblock.glassfree3d.ducks.MinecraftAccessor;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Mutable;
import org.spongepowered.asm.mixin.Shadow;

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
}
