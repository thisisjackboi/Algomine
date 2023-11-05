from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib to use the Agg backend

import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

import networkx as nx
from sklearn.cluster import KMeans

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/algo.html')
def algo():
    return render_template('algo.html')


@app.route('/algo_apriori.html', methods=["GET", "POST"])
def algo_apriori():
    
    recommendations = []
        
    form_submitted = False  # Initialize the form submission status

    if request.method == "POST":
        form_submitted = True  # Set the form submission status to True
        # Get the minimum support and minimum confidence values from the form
        min_support = float(request.form.get("min_support", 0.0))
        min_confidence = float(request.form.get("min_confidence", 0.0))
        
        
        
        # Load the data from 'countries_exercise.csv'
        data = pd.read_csv('static/data/groceries_dataset.csv')

   

        basket = data
        
        basket.itemDescription = basket.itemDescription.transform(lambda x: [x])
        basket = basket.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)

        encoder = TransactionEncoder()
        transactions = pd.DataFrame(encoder.fit(basket).transform(basket), columns=encoder.columns_)
        
        frequent_itemsets = apriori(transactions, min_support= 6/len(basket), use_colnames=True, max_len = 2)
        rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5)
     
        print("Rules identified: ", len(rules))
        
        # Check if rules were generated
        if not rules.empty:
            # Extract high-lift rules
            high_lift_rules = rules[rules['lift'] > 1.0]

            # Display the high-lift rules
            

            # Generate recommendations based on high-lift rules
            for _, row in high_lift_rules.iterrows():
                antecedents = ', '.join(row['antecedents'])
                consequents = ', '.join(row['consequents'])
                lift = row['lift']

                recommendation = f"When customers buy {antecedents}, recommend {consequents} (Lift: {lift:.2f})"
                
        else:
            print("No association rules found.")    
           
        imagepath = draw_network(rules, 6)

        
        return render_template('image.html', image=imagepath)

    return render_template("algo_apriori.html", form_submitted=form_submitted)

def draw_network(rules, rules_to_show):
    # Create a directional graph using NetworkX
    network = nx.DiGraph()

    # Define a color map to distinguish rules (black) from other nodes (orange)
    color_map = {}

    for i in range(rules_to_show):
        rule_name = f"R{i}"
        antecedents = rules.iloc[i]['antecedents']
        consequents = rules.iloc[i]['consequents']

        # Add a rule node and set its color to black
        network.add_node(rule_name)
        color_map[rule_name] = 'black'

        # Add antecedent nodes and edges
        for antecedent in antecedents:
            network.add_node(antecedent)
            network.add_edge(antecedent, rule_name, weight=2)
            color_map[antecedent] = 'orange'

        # Add consequent nodes and edges
        for consequent in consequents:
            network.add_node(consequent)
            network.add_edge(rule_name, consequent, weight=2)
            color_map[consequent] = 'orange'

    # Position nodes using the spring layout algorithm
    pos = nx.spring_layout(network, k=16, scale=1)

    # Shift the text position upwards to avoid overlap with nodes
    for p in pos:
        pos[p][1] += 0.12

    # Draw the network graph with node colors and labels
    nx.draw(network, pos, node_color=list(color_map.values()), font_size=8, labels={node: node for node in network.nodes})

    # Set the title for the network graph
    plt.title("Network Graph for Association Rules")

    # Save the network graph as an image
    imagepath = os.path.join(app.config['UPLOAD_FOLDER'], 'img', 'image.png')
    plt.savefig(imagepath)
    plt.close()
    
    return imagepath



@app.route('/algo_kmeans.html', methods=["GET", "POST"])
def algo_kmeans():
    if request.method == "POST":
        # Get the user's input for K from the form
        k_value = int(request.form.get('k_value'))

        # Load the data from 'countries_exercise.csv'
        data = pd.read_csv('static/data/countries_exercise.csv')

        # Extract the features (longitude and latitude)
        x = data.iloc[:, 1:3]

        # Create a KMeans model with the user-provided K value
        kmeans = KMeans(n_clusters=k_value)

        # Fit the KMeans model to the data and predict cluster labels
        identified_clusters = kmeans.fit_predict(x)

        # Define cluster descriptions
        cluster_descriptions = {}
        for i in range(k_value):
            cluster_descriptions[i] = f'Cluster {i}'

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_

        # Create a copy of the original data with the cluster labels
        data_with_clusters = data.copy()
        data_with_clusters['Clusters'] = identified_clusters

        # Create a scatter plot
        plt.figure(figsize=(10, 6))

        # Scatter plot of data points with cluster colors and descriptions in the legend
        #just checking
        for cluster_label, description in cluster_descriptions.items():
            cluster_data = data_with_clusters[data_with_clusters['Clusters'] == cluster_label]
            plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f'{description}')

        # Scatter plot of centroids with black color
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=100, c='black', label='Centroids')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'KMeans Clustering of Countries with K={k_value}')
        plt.legend(title='Cluster')
        plt.grid(True)

        # Save the plot as 'kimage.png' in the 'static/img' directory
        plt.savefig('static/img/kimage.png')
        plt.close()  # Close the plot
        return render_template('imagek.html')

    return render_template('algo_kmeans.html')
    

        
if __name__ == '__main__':
    app.run(debug=True)


@app.errorhandler(404)
def page_not_found(e):
    return "404 - Page Not Found-> KIBA ETA GORBOR ASE", 404


if __name__ == '__main__':
    app.run(debug=True)