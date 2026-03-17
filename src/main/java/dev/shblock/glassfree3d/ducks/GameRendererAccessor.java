package dev.shblock.glassfree3d.ducks;

import net.minecraft.world.phys.HitResult;

import java.util.function.Supplier;

public interface GameRendererAccessor {
    void gf_addPicker(Supplier<HitResult> picker);
}
