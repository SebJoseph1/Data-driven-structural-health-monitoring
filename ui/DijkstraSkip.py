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
        """ 
        Function that creates all the edges connected to pixels (nodes) outside the stripe (crack)
        input: 
        k is the current depth of the grid
        startpoints is the list of nodes from which connections should be made
        img is the image on which checker should work
        g is the network to which the edges are added
        output:
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
        g = nx.Graph()
        start_time = time.time() 
        # Find all pixels that are white and thus are recognized to be part of the crack 
        indices = np.argwhere(img == 255)
        # Check if there is a crack, if not return that the path has length 0 and the original image
        if len(indices) == 0:
            return (0, img,0,0)

        minrow = min(indices[:,0])
        maxrow = max(indices[:,0])
        mincol = min(indices[:,1])
        maxcol = max(indices[:,1])

        # Count black rows
        blackrows = 0
        maxblackrows = 0
        for idx in range(minrow, maxrow+1):
            lst = img[idx].tolist()
            if lst.count(255) == 0:
                blackrows += 1
            else:
                if blackrows > maxblackrows:
                    maxblackrows = blackrows    
                blackrows = 0

        # Count black columns
        blackcols = 0
        maxblackcols = 0
        for idx in range(mincol, maxcol+1):
            lst = img[:,idx].tolist()
            if lst.count(255) == 0:
                blackcols += 1
            else:
                if blackcols > maxblackcols:
                    maxblackcols = blackcols    
                blackcols = 0
        
        if max(maxblackcols, maxblackrows) > 30:
            print("Dijkstra didn't work due to too much seperate label with cols ",maxblackcols," and rows ",maxblackrows)
            return (0, img, 0, 0)

        # Calculate how deep the grid must be to handle the desired gapwidth
        if GAPWIDTH == 0:
            blackdepth = 0
        elif GAPWIDTH > 0:
            blackdepth = int(GAPWIDTH/2)+1
        else:
            raise ValueError('Choose a non-negative value for the GAPWIDTH')


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

        # Find the start and end points of the stripe
        for i in range(img.shape[1]):
            column_values = np.where(img[:, i] == 255)[0]
            if len(column_values) > 0:
                start = (column_values[0], i)
                break   

        for i in range(img.shape[1]-1, -1, -1):
            column_values = np.where(img[:, i] == 255)[0]
            if len(column_values) > 0:
                end = (column_values[0], i)
                break
        # Efficient built-in algorithm that determines the shortest path from the determined start and end point
        path = nx.dijkstra_path(g, start, end)
        
        # Find the start and end points of the stripe
        for i in range(img.shape[1]):
            column_values = np.where(img[:, i] == 255)[0]
            if len(column_values) > 0:
                start = (column_values[0], i)
                break   

        for i in range(img.shape[1]-1, -1, -1):
            column_values = np.where(img[:, i] == 255)[0]
            if len(column_values) > 0:
                end = (column_values[0], i)
                break

        # Output the length of the path in pixels
        print(f"Length of stripe: {len(path)}")
        # Output the computation time
        print("--- %s seconds ---" % ((time.time() - start_time)))

        # Colour all pixels that are part of the path red
        img_path = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in path:
            img_path[point[0], point[1]] = (0, 0, 255)      # Red colour (BGR format)

        return(len(path),img_path,start,end)

