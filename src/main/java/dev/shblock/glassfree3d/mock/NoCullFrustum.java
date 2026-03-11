package dev.shblock.glassfree3d.mock;

import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.world.phys.AABB;
import org.joml.Matrix4f;

public class NoCullFrustum extends Frustum {
    public NoCullFrustum() {
        super(new Matrix4f(), new Matrix4f());
    }

    @Override
    public Frustum offsetToFullyIncludeCameraCube(int offset) {
        return this;
    }

    @Override
    public boolean isVisible(AABB aabb) {
        return true;
    }
}
