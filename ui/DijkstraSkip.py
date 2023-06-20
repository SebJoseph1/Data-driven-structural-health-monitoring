import numpy as np
import cv2
import networkx as nx
import time
import math

class Dijkstra():

    def __init__(self,gapwidth):
        self.gapwidth = gapwidth

    @staticmethod
    def checker(k, startpoints,img,g):
        """ Function that creates all the edges connected to pixels (nodes) outside the stripe (crack)
        k is the current depth of the grid
        startpoints is the list of nodes from which connections should be made
        bl returns all the future startpoints (with duplicates)
        """

        bl = []
        for sta in startpoints:
            black = []
            x = sta[0]
            y = sta[1]
            if x > 0 and img[x-1, y] != 255:                # Check node above
                black.append((x-1, y))
            if x < img.shape[0]-1 and img[x+1, y] != 255:   # Check node below
                black.append((x+1, y))
            if y > 0 and img[x, y-1] != 255:                # Check node at the left
                black.append((x, y-1))
            if y < img.shape[1]-1 and img[x, y+1] != 255:   # Check node at the right
                black.append((x, y+1))
            for b in black:
                g.add_edge((x, y), b, weight=10**(k+2))     # Add all edges with a weight depended on the depth of the grid
            bl.extend(black)    
        return bl
    
    def predict(self,filepath):
        GAPWIDTH = self.gapwidth
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # Start recording time of path determination
        start_time = time.time() 

        # Initialize the graph
        g = nx.Graph()

        # Calculate how deep the grid must be to handle the desired gapwidth
        if GAPWIDTH == 0:
            blackdepth = 0
        elif GAPWIDTH > 0:
            blackdepth = int(GAPWIDTH/2)+1
        else:
            raise ValueError('Choose a non-negative value for the GAPWIDTH')

        # Find all pixels that are white and thus are recognized to be part of the crack 
        indices = np.argwhere(img == 255)

        # Check if there is a crack, if not return that the path has length 0 and the original image
        if len(indices) == 0:
            return ([0], img,[(0,0)],[(0,0)])

        # Split the data of the cracks in the image
        sorted_indices = indices[np.argsort(indices[:, 1])]
        start = sorted_indices[0]                                   # Use the left most white pixel in the image as the starting point for the first crack
        crack_nr = 1
        crack0 = []
        newcrack = []
        while len(indices) > 0:                                     # Keep making new cracks until all white pixels are assigned to a crack
            distances = np.linalg.norm(indices - start, axis=1)     # Calculate for every pixel still unassigned its distance to the start pixel
            othercrack = []
            i = 0
            for distance in distances:                              # Add the pixels that are less than GAPWIDTH away to the current crack, keep the others
                if distance < GAPWIDTH:                 
                    crack0.append(indices[i].tolist())
                    newcrack.append(indices[i].tolist())            # To check whether new pixels are being added or a new crack should be started
                else:
                    othercrack.append(indices[i].tolist())
                i += 1 
            if newcrack:                                            # If there are still pixels being added to the crack, continue with this crack
                cr0 = np.array(crack0, dtype=np.int64)
                sorted_crack0 = cr0[np.argsort(cr0[:, 1])]          # Sort the indices in ascending order based on the column values
                start = sorted_crack0[-1]                           # Use the rightmost pixel that is now part of the crack as the start pixel to find more pixels
            indices = np.array(othercrack, dtype=np.int64)
            if not newcrack:                                        # If no new pixel are being added to the crack, a new crack should be started
                exec(f"crack{crack_nr} = crack0")
                crack0 = []
                crack_nr += 1
                othercrack = []
                sorted_indices = indices[np.argsort(indices[:, 1])] # Sort the still unassigned pixels in ascending order based on column values
                start = sorted_indices[0]                           # Use the left most of the pixels as the start of the new crack
            newcrack = []

        start_list = []
        end_list = []
        crack_length_list = []
        # Loop through all the defined cracks to determine their length
        for nr in range(crack_nr):                                  
            crackvalues = eval(f"crack{nr}")
            indices = np.array(crackvalues, dtype=np.int64)
            sorted_indices = indices[np.argsort(indices[:, 1])]
            start = tuple(sorted_indices[0])                        # Recognize the start of the current crack
            end = tuple(sorted_indices[-1])  
            start_list.append(start)
            end_list.append(end)                       # Recognize the end of the current crack
            # A for loop that considers all the pixels recognized to be part of the crack as nodes, and creates edges
            for idx in indices:
                i = idx[0]
                j = idx[1]

                # Create all edges between the node (i,j) and neighboring white pixels
                neighbors = []
                if i > 0 and img[i-1, j] == 255:                # Check node above
                    neighbors.append((i-1, j))
                if i < img.shape[0]-1 and img[i+1, j] == 255:   # Check node below 
                    neighbors.append((i+1, j))
                if j > 0 and img[i, j-1] == 255:                # Check node at the left
                    neighbors.append((i, j-1))
                if j < img.shape[1]-1 and img[i, j+1] == 255:   # Check node at the right
                    neighbors.append((i, j+1))
                for neighbor in neighbors:
                    g.add_edge((i, j), neighbor, weight=1)      # Add the edges with a weight of 1, meaning that they are preferred for the path

                # Create all edges with black pixels to a desired depth
                startpoints = [(i, j)]
                for k in range(blackdepth):                     # Loops through the depth of the grid s.t. the edges are added to the network
                    startpoints = self.checker(k, startpoints,img,g)       # Use the pixels to which the edges are created as startpoints for the next iteration
                    startpoints = list(set(startpoints))        # Remove duplicates to increase speed and prevent double edges

            # Efficient built-in algorithm that determines the shortest path from the determined start and end point
            path = nx.dijkstra_path(g, start, end)

            # Output the length of the path in pixels
            # print(f"Length of stripe: {len(path)}")
            # Output the computation time
            # print("--- %s seconds ---" % ((time.time() - start_time)))
            crack_length_list.append(len(path))
            # Colour all pixels that are part of the path red
            img_path = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for point in path:
                img_path[point[0], point[1]] = (0, 0, 255)      # Red colour (BGR format)



        return(crack_length_list,img_path,start_list,end_list)

