plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)

# Add a color bar to show what pressure values the colors correspond to
plt.colorbar(label='Pressure')

# --- 3. Plot the velocity field ---
# Draw streamlines to show the direction of the flow (u, v)
# The density parameter controls how many lines are drawn.
plt.streamplot(X, Y, u, v, color='black', density=1.2)
# An alternative is a quiver plot, which draws arrows:
# plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])

# --- 4. Final touches ---
plt.title('Lid-Driven Cavity Flow')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the plot
plt.show()