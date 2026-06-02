import open_clip
tags = open_clip.list_pretrained()
for arch in ["ViT-L-14","ViT-B-16","ViT-B-32","convnext_large_d","convnext_base_w","ViT-L-14-quickgelu"]:
    pts = [t[1] for t in tags if t[0]==arch]
    print(arch, "=>", pts)
