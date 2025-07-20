package dev.shblock.glassfree3d.mixin;

import net.minecraft.client.renderer.culling.Frustum;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(Frustum.class)
public class FrustumMixin {
    @Inject(
        method = "offsetToFullyIncludeCameraCube",
        at = @At("HEAD"),
        cancellable = true
    )
    void fixOffsetToFullyIncludeCameraCubeForOffAxis(int offset, CallbackInfoReturnable<Frustum> cir) {
        cir.setReturnValue((Frustum) (Object) this);
        cir.cancel();
    }
}

