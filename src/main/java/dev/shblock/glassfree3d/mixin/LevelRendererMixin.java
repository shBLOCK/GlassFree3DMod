package dev.shblock.glassfree3d.mixin;

import dev.shblock.glassfree3d.ducks.LevelRendererAccessor;
import dev.shblock.glassfree3d.mock.NoCullFrustum;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Constant;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyConstant;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(LevelRenderer.class)
public abstract class LevelRendererMixin implements LevelRendererAccessor {
    @Unique
    public boolean disableFrustumCulling = false;

    @Override
    public void gf_setDisableFrustumCulling(boolean value) {
        disableFrustumCulling = value;
    }

    @Shadow private Frustum cullingFrustum;
//    @Shadow @Final private Minecraft minecraft;
//    @Shadow @Final public ObjectArrayList<SectionRenderDispatcher.RenderSection> visibleSections;

//    @Shadow @Final public SectionOcclusionGraph sectionOcclusionGraph;
//    @Overwrite
//    private void applyFrustum(Frustum frustum) {
//        if (!Minecraft.getInstance().isSameThread()) {
//            throw new IllegalStateException("applyFrustum called from wrong thread: " + Thread.currentThread().getName());
//        } else {

//            this.minecraft.getProfiler().push("apply_frustum");

    ////            this.visibleSections.clear();
//            this.sectionOcclusionGraph.addSectionsInFrustum(frustum, this.visibleSections);
//            this.minecraft.getProfiler().pop();
//        }
//    }

    @Inject(method = "prepareCullFrustum", at = @At("HEAD"), cancellable = true)
    private void prepareCullFrustum(Vec3 cameraPosition, Matrix4f frustumMatrix, Matrix4f projectionMatrix, CallbackInfo ci) {
        if (disableFrustumCulling) {
            this.cullingFrustum = new NoCullFrustum();
            this.cullingFrustum.prepare(cameraPosition.x, cameraPosition.y, cameraPosition.z);
            ci.cancel();
        }
    }

    @Inject(method = "offsetFrustum", at = @At("HEAD"), cancellable = true)
    private static void offsetFrustum(Frustum frustum, CallbackInfoReturnable<Frustum> cir) {
        if (frustum instanceof NoCullFrustum) {
            cir.setReturnValue(frustum);
        }
    }
    
    @ModifyConstant(method = "renderLevel", constant = @Constant(doubleValue = 1024.0F), allow = 1)
    private double modifyBlockDestructionProgressRenderRange(double _value) {
        return Float.POSITIVE_INFINITY;
    }
}
