
datafile = "/orange/pinaki.sarder/ahmed.naglah/inflammation_project/inflammatory_cell_density_r.csv"

data = read.csv(datafile)

library(ggplot2)
# ggplot(data, aes(x=feature, y=value)) + geom_violin()


# p <- ggplot(data, aes(Tissue.Region, Inflammatory.Cells.Density)) + 
#   geom_boxplot() +
#   geom_point() +
#   geom_line(aes(group = Sample)) +
#   scale_x_discrete(limits = c('Inflammation Regions', 'Non-Inflammation Regions'))

# p

p <- ggplot(data, aes(x = Tissue.Region, y = Inflammatory.Cells.Density, fill = Tissue.Region)) + 
  geom_boxplot(outlier.shape = NA) +
  geom_point(color = "black", size = 4, position = position_jitter(width = 0.2)) +
  scale_fill_brewer(palette = "Blues") + 
  labs(x = "Ground Truth", y = expression ("Predicted Inflammatory Cell Density ("~cells/mm^2~")")) + 
  theme(
    panel.background = element_rect(fill = "white"),     # White background
    plot.title = element_text(size = 28, face = "bold"), 
    panel.grid.major = element_line(color = "gray", size = 0.25), # Minor grid lines in black
    axis.title.x = element_text(size = 20),             
    axis.title.y = element_text(size = 20),             
    axis.text = element_text(size = 18),    
    legend.position = "none"
  )

p

ggsave(file="/blue/pinaki.sarder/ahmed.naglah/experiments/inflammation/r_results/approach01.svg", plot=p, width=10, height=9)





datafile = "/orange/pinaki.sarder/ahmed.naglah/inflammation_project/inflammation_all_cells.csv"

data = read.csv(datafile)

library(ggplot2)
# ggplot(data, aes(x=feature, y=value)) + geom_violin()


p <- ggplot(data, aes(feature, inflammatory.cells.density)) + 
  geom_boxplot() +
  geom_point() +
  geom_line(aes(group = sample)) +
  scale_x_discrete(limits = c('inside.lesion', 'outside.lesion'))

p

p <- ggplot(data, aes(feature, inflammatory.cells.density)) + 
  geom_boxplot() +
  geom_point(color="gray", size=5) +
  scale_fill_brewer(palette="Blues") + 
  theme(
    plot.title = element_text(size = 28, face = "bold"), # Title font size
    axis.title.x = element_text(size = 20),             # X-axis title font size
    axis.title.y = element_text(size = 20),             # Y-axis title font size
    axis.text = element_text(size = 18),                # Axis text (ticks) font size
    legend.text = element_text(size = 18),              # Legend text font size
    legend.title = element_text(size = 18)              # Legend title font size
  )

p

ggsave(file="/blue/pinaki.sarder/ahmed.naglah/experiments/inflammation/r_results/approach01.svg", plot=p, width=12, height=10)