

1. 数据处理部分，要对Bioassembly dict进行处理，导入的结构pdb 或cif，最好是对应的antibody 和antigen -complex，不会有其他的链
2. indcies csv 更新的时候应该添加 antibody chain ids ？
3. 模型主框架部分，添加了seq denoised （AtomDiffusion），loss  部分添加了这部分的seq loss 计算逻辑
4， 