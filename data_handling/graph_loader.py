import networkx as nx
import numpy as np
import os
from typing import Dict, Tuple, List, Any, Optional
import logging

logger = logging.getLogger(__name__) 

DEFAULT_NODES_FILE = "nodes.txt"
DEFAULT_LINKS_FILE = "links.txt" 
DEFAULT_IMAGES_FILE = "images.txt" 

class VLNGraphLoader:
    """
    Loads navigation graph data for VLN datasets (e.g., Touchdown, Map2Seq).
    It expects graph data typically stored in text files defining nodes and their connectivity.
    The output is a NetworkX graph and a dictionary of node positions.
    """
    def __init__(self, graph_data_dir: str,
                 nodes_filename: str = DEFAULT_NODES_FILE,
                 links_filename: str = DEFAULT_LINKS_FILE,
                 images_filename: Optional[str] = DEFAULT_IMAGES_FILE):
        """
        Initializes the graph loader.

        Args:
            graph_data_dir (str): The directory containing the graph data files
                                  (e.g., 'nodes.txt', 'links.txt').
            nodes_filename (str): Name of the file containing node information.
            links_filename (str): Name of the file containing edge/link information.
            images_filename (Optional[str]): Name of the file mapping panoids to image files (if available).
        """
        self.graph_data_dir = graph_data_dir
        self.nodes_file = os.path.join(graph_data_dir, nodes_filename)
        self.links_file = os.path.join(graph_data_dir, links_filename)
        self.images_file = os.path.join(graph_data_dir, images_filename) if images_filename else None

        self.graph: Optional[nx.Graph] = None
        self.node_positions: Dict[str, np.ndarray] = {} # panoid -> [x,y,z]
        self.node_headings: Dict[str, float] = {} # panoid -> main heading (if available)
        self.pano_to_image_file: Dict[str, str] = {} # panoid -> image_filename

        if not os.path.isdir(graph_data_dir):
            logger.warning(f"Graph data directory not found: {graph_data_dir}. Graph will be empty.")
            self.graph = nx.Graph() # Initialize with an empty graph
            return

        self._load_graph_data()

    def _load_nodes(self):
        """
        Loads node data (panoid, position, heading) from the nodes file.
        File format example (tab-separated):
        pano_id  easting_m  northing_m  height_m  heading_deg  covered_dist_m
        gsv_pano_000000 327628.31 4518198.91 19.01 110.81 0.00
        """
        if not os.path.exists(self.nodes_file):
            logger.warning(f"Nodes file not found: {self.nodes_file}. Node positions and headings will be empty.")
            return

        with open(self.nodes_file, 'r') as f:
            data_to_read = f.readlines()
        for line in data_to_read:
            parts = line.strip().split('\t')
            if not parts or len(parts) < 5: # Expect at least pano_id, x, y, z, heading
                # logger.debug(f"Skipping malformed node line: {line.strip()}")
                continue
            
            panoid = parts[0]
            try:
                # Easting (x), Northing (y), Height (z) - standard UTM order for StreetLearn data
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                heading_deg = float(parts[4]) # Main heading of the panorama
                
                self.node_positions[panoid] = np.array([x, y, z], dtype=np.float32)
                self.node_headings[panoid] = heading_deg
                if self.graph is not None:
                    self.graph.add_node(panoid, pos=self.node_positions[panoid], heading=heading_deg)
            except ValueError as e:
                logger.warning(f"Skipping node line due to parsing error: {line.strip()} - {e}")


    def _load_links(self):
        """
        Loads link/edge data from the links file and adds edges to the graph.
        File format example (tab-separated):
        from_pano_id  to_pano_id  heading_deg  length_m
        gsv_pano_000000 gsv_pano_000001 110.81 10.00
        """
        if not os.path.exists(self.links_file):
            logger.warning(f"Links file not found: {self.links_file}. Graph may have no edges.")
            return
        if self.graph is None: # Should have been initialized
            self.graph = nx.Graph()


        with open(self.links_file, 'r') as f:
            data_to_read = f.readlines()
        for line in data_to_read:
            parts = line.strip().split('\t')
            if not parts or len(parts) < 4: 
                # logger.debug(f"Skipping malformed link line: {line.strip()}")
                continue

            from_panoid = parts[0]
            to_panoid = parts[1]
            try:
                heading_deg = float(parts[2]) 
                length_m = float(parts[3])    

                if from_panoid not in self.graph: # type: ignore
                    self.graph.add_node(from_panoid) # type: ignore
                if to_panoid not in self.graph: # type: ignore
                    self.graph.add_node(to_panoid) # type: ignore

                self.graph.add_edge(from_panoid, to_panoid, weight=length_m, heading=heading_deg) # type: ignore

            except ValueError as e:
                logger.warning(f"Skipping link line due to parsing error: {line.strip()} - {e}")

    def _load_image_mappings(self):
        """
        Loads mappings from panoid to actual image filenames if available.
        File format example (tab-separated):
        pano_id  image_file_name.jpg
        """
        if self.images_file and os.path.exists(self.images_file):
            with open(self.images_file, 'r') as f:
                data_to_read = f.readlines()
            for line in data_to_read:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.pano_to_image_file[parts[0]] = parts[1]
        else:
            logger.info(f"Images mapping file not found or not specified: {self.images_file}. Pano to image file mapping will be empty.")


    def _load_graph_data(self):
        """Loads all graph components: nodes, then links, then image mappings."""
        self.graph = nx.Graph() # Initialize an empty graph
        
        logger.info(f"Loading graph data from directory: {self.graph_data_dir}")
        self._load_nodes()
        self._load_links() 
        self._load_image_mappings()

        if self.graph and self.graph.number_of_nodes() > 0:
            logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        else:
            logger.warning("Graph loading resulted in an empty or uninitialized graph.")


    def get_graph(self) -> Optional[nx.Graph]:
        """Returns the loaded NetworkX graph."""
        return self.graph

    def get_node_positions(self) -> Dict[str, np.ndarray]:
        """Returns a dictionary mapping panoid to its [x,y,z] position."""
        return self.node_positions

    def get_node_headings(self) -> Dict[str, float]:
        """Returns a dictionary mapping panoid to its primary heading in degrees."""
        return self.node_headings
        
    def get_pano_to_image_file_mapping(self) -> Dict[str, str]:
        """Returns a dictionary mapping panoid to its image filename."""
        return self.pano_to_image_file


if __name__ == '__main__':
    print("--- Testing VLNGraphLoader ---")

    # Create dummy graph files for testing
    dummy_graph_dir = "./tmp_test_graph_data"
    os.makedirs(dummy_graph_dir, exist_ok=True)

    # Dummy nodes.txt
    nodes_content = (
        "pano_A\t0.0\t0.0\t0.0\t0.0\t0.0\n"
        "pano_B\t10.0\t0.0\t0.0\t0.0\t10.0\n"
        "pano_C\t10.0\t10.0\t0.0\t90.0\t20.0\n"
        "pano_D\t0.0\t10.0\t0.0\t180.0\t30.0\n" # Extra node not in links
        "pano_E\t-10.0\t0.0\t0.0\t270.0\t0.0\n" # Node only in links
        "malformed_node\ttext\n"
    )
    with open(os.path.join(dummy_graph_dir, DEFAULT_NODES_FILE), "w") as f:
        f.write(nodes_content)

    # Dummy links.txt
    links_content = (
        "pano_A\tpano_B\t0.0\t10.0\n"
        "pano_B\tpano_C\t90.0\t10.0\n"
        "pano_C\tpano_A\t225.0\t14.14\n" # A->B->C->A triangle
        "pano_A\tpano_E\t270.0\t10.0\n" # Link to a node not in nodes.txt initially
        "malformed_link\ttext\n"
    )
    with open(os.path.join(dummy_graph_dir, DEFAULT_LINKS_FILE), "w") as f:
        f.write(links_content)

    # Dummy images.txt
    images_content = (
        "pano_A\timage_A.jpg\n"
        "pano_B\timage_B.jpg\n"
    )
    with open(os.path.join(dummy_graph_dir, DEFAULT_IMAGES_FILE), "w") as f:
        f.write(images_content)


    # Test loading
    print(f"\nLoading graph from: {dummy_graph_dir}")
    # Setup a basic logger for the test if not already configured by main
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    graph_loader = VLNGraphLoader(dummy_graph_dir)
    
    loaded_graph = graph_loader.get_graph()
    node_positions = graph_loader.get_node_positions()
    node_headings = graph_loader.get_node_headings()
    image_map = graph_loader.get_pano_to_image_file_mapping()

    if loaded_graph:
        print(f"  Graph loaded successfully: {loaded_graph.number_of_nodes()} nodes, {loaded_graph.number_of_edges()} edges.")
        assert loaded_graph.number_of_nodes() == 5 # A, B, C, D, E (D from nodes, E from links)
        assert loaded_graph.number_of_edges() == 4 
        

        assert "pano_A" in node_positions and np.allclose(node_positions["pano_A"], [0,0,0])
        assert "pano_D" in node_positions 
        assert "pano_E" in loaded_graph.nodes() 
        assert "pano_A" in node_headings and np.isclose(node_headings["pano_A"], 0.0)

  
        if loaded_graph.has_edge("pano_A", "pano_B"):
            edge_data_AB = loaded_graph.get_edge_data("pano_A", "pano_B")
            print(f"  Edge A-B data: {edge_data_AB}")
            assert np.isclose(edge_data_AB.get('weight', -1), 10.0)
            assert np.isclose(edge_data_AB.get('heading', -1), 0.0)
        else:
            print("  ERROR: Edge A-B not found.")
        # Check image map
        print(f"  Image map: {image_map}")
        assert image_map.get("pano_A") == "image_A.jpg"
        assert len(image_map) == 2

    else:
        print("  ERROR: Graph loading failed or resulted in an empty graph.")

    import shutil
    shutil.rmtree(dummy_graph_dir)
    print("\nVLNGraphLoader tests completed and cleanup done.")