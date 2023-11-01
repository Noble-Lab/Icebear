args = commandArgs(trailingOnly=TRUE)
library(data.table)
library(ggplot2)

sim_url <- args[1]
if (file.exists(paste0(sim_url, '_umap.txt'))){
  embedding_mat <- fread(paste0(sim_url, '_umap.txt'), sep='\t', header=F)
  names(embedding_mat) <- c('UMAP_1', 'UMAP_2','batch','celltype', 'species')
  embedding_mat <- as.data.frame(embedding_mat)
  #embedding_mat$dataset <- factor(embedding_mat$dataset)
  embedding_mat$celltype <- factor(embedding_mat$celltype)
  embedding_mat$batch <- factor(embedding_mat$batch)
  embedding_mat$species <- factor(embedding_mat$species)
  
  ## plot all, separation by batch, time or cell type

  p1_pred_combined_umap_ct <- ggplot(embedding_mat, aes(UMAP_1, UMAP_2, color = species))+ facet_wrap(~celltype) + 
    theme_classic() + theme(panel.background = element_rect(fill = 'white', colour = 'white'), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) + theme(legend.position = "none")+
    geom_point(size=1, alpha = 0.2)
  
  png(paste0(sim_url, '_umap_species.png'), width = 1000, height = 800, res=110)
  print(p1_pred_combined_umap_ct)
  dev.off()
  
  p1_pred_combined_umap_species <- ggplot(embedding_mat, aes(UMAP_1, UMAP_2, color = batch))+ facet_wrap(~species) + 
    theme_classic() + theme(panel.background = element_rect(fill = 'white', colour = 'white'), panel.border = element_rect(colour = "black", fill=NA, size=0.8)) + theme(legend.position = "none")+
    geom_point(size=1, alpha = 0.2)
  
  png(paste0(sim_url, '_umap_celltype.png'), width = 1100, height = 400, res=110)
  print(p1_pred_combined_umap_species)
  dev.off()
}

