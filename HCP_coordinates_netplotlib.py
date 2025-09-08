
# Load communities
betweenness_filepath = r"C:\Users\DELL 7560\Desktop\as_betweenness.csv"
betweeness_df = pd.read_csv(betweenness_filepath)

vals = list(betweeness_df["pval"])

com = []
for i in vals:
    if i <= 0.05:
        com.append(1)
    else:
        com.append(0)

# Set up HCP filepath
hcp_filepath = "HCP_coordinates.csv"
file_raw_data = pd.read_csv(hcp_filepath)

xyz_dataframe = pd.DataFrame({'x': list(file_raw_data["z-cog"]-100),
                              'y': list(file_raw_data["y-cog"]-135),
                              'z': list(file_raw_data["x-cog"]-80),
                              "communities": com})

nodes = xyz_dataframe

fig, ax = netplotbrain.plot(template='MNI152NLin2009cAsym',
                            nodes=nodes,
                            edges=diff_array,
                            node_color='communities',
                            highlight_edges=adj,
                            template_style='glass',
                            view='LSR',
                            title='NBS Integration',
                            node_type='circles',
                            highlight_level=1)
plt.show()
