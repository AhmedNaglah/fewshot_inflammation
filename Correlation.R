
datafile = "/blue/pinaki.sarder/ahmed.naglah/experiments/inflammation/inflammation_ai.csv"

data = read.csv(datafile)

model <- lm(ai~gt, data = data)

# Display model summary
summary(model)

r_squared <- summary(model)$r.squared

# Extract p-value
p_value <- summary(model)$coefficients[2, 4]

cat("R-squared:", r_squared, "\nP-value:", p_value)

png("plot_high_res.png", width = 2000, height = 1500, res = 300) # Adjust dimensions as needed

plot(data$gt*100, data$ai*100, 
     main = "Predicted versus True inflammatory load",
     xlab = "Total Inflammation % (True)", 
     ylab = "Total Inflammation % (Predicted)",
     pch = 19, # Filled circle
     col = "black",
     cex.axis = 1.4,  # Increases font size of tick labels
     cex.lab = 1.2,
     cex = 1.9,
     cex.main = 1.5)

# Add regression line
abline(model, col = "orange", lwd = 3, lty = 4)

# Annotate the R-squared value on the plot
r_squared <- summary(model)$r.squared
p_value_model <- summary(model)$fstatistic
sig <- pf(p_value_model[1], p_value_model[2], p_value_model[3], lower.tail = FALSE)

r_squared_formatted <- formatC(r_squared, format = "f", digits = 2)
sig_formatted <- formatC(sig, format = "f", digits = 2)

legend("topleft", 
       legend = paste("R-squared =", r_squared_formatted, "   P-Value = ", sig_formatted), 
       col = "gray", 
       lty = 1, 
       bty = "n",
       cex = 1.5) # No box around legend

dev.off()  # Close the graphics device

# Add text labels for each point
text(data$gt, data$ai, 
     labels = data$participant, 
     pos = 1, # Position: 4 = right of the point
     col = "black", 
     cex = 2) # Text size










library(ggplot2)

datafile = "/blue/pinaki.sarder/ahmed.naglah/experiments/inflammation/inflammation_ai.csv"

data = read.csv(datafile)


# Scatter plot with regression line and R^2
p <- ggplot(data, aes(x = gt, y = ai)) + 
  geom_point(color = "black", size = 3) +  # Scatter points
  scale_x_continuous(limits = c(0, 100)) +   # Set x-axis limits
  scale_y_continuous(limits = c(0, 100)) +   # Set y-axis limits
  geom_smooth(method = "lm", formula = y ~ x, color = "orange", size = 1, linetype = "dashed") +  # Regression line
  labs(
    x = "Total Inflammation % (True)",
    y = "Total Inflammation % (Predicted)"
  ) +
  # Add custom theme
  theme(
    panel.background = element_rect(fill = "white"),     # White background
    panel.grid.major = element_line(color = "gray", size = 0.25),  # Black major grid lines
    plot.title = element_text(size = 28, face = "bold"), 
    axis.title.x = element_text(size = 20),             
    axis.title.y = element_text(size = 20),             
    axis.text = element_text(size = 18),                
    legend.position = "none"                            
  ) +
  # Display R^2
  annotate("text", x = min(data$gt), y = max(data$ai), 
           label = paste("R-Squared= ", round(summary(lm(ai ~ gt, data = data))$r.squared, 2), "  P-value = 0.001"), 
           size = 6, hjust = 0)

print(p)

ggsave(file="/blue/pinaki.sarder/ahmed.naglah/experiments/inflammation/r_results/approach02_Updated.svg", plot=p, width=12, height=12)
