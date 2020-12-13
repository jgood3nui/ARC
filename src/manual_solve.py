#!/usr/bin/python
# https://github.com/jgood3nui/ARC.git
# Student Name: Jonathan Good
# Student Number: 20235990

import os, sys
import json
import numpy as np
import re
import itertools
from math import sqrt
from sklearn.linear_model import LinearRegression

'''
Summary/Reflection

For a number of these task I have tried to optimize them for generality 
and flexibility in line with the spirit of Chollet's paper "On the Measure of 
Intelligence". Generalization as he outlines is the ability to handle tasks 
that differ from the previous. While a machine learning program can be excellent 
at chess, that doesn't mean it could even perform well at another game like Go. 
Intelligence would then be skill (ability to win at chess) plus adapability 
(ability to win at any game without just adding more training data). While 
hard coding a solution is obviously not intelligence neither is simpley adding more 
training data.

All the solutions use a Scikit-learn (open source machine learning library)
linear regression model to predict the solution's final shape based on the 
training examples and many utilise the convert_shape function. ac0a08a4 and b91ae062 have the same code bar the 
task id passed to the regression model for getting the training examples. 

These solutions also heavily make us of numpy arrays. The numpy library comes
with many useful functions such finding the unique values (.unique), deleting rows (.delete),
finding values (.where) and rotating the grid (.rot90).

Itertools allows combinding lists to get each possible been them and saves on createing
for loops within for loops. The library also contains combinations which produces 
all possible combinations of its elements of a given length

'''

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_681b3aeb(x):
    """
    # Solves the task 681b3aeb. For this task the intial grid contains two
    coloured shapes with. The solution requires finding the two shapes and
    fitting them together to form a smaller grid. 
    
    As Chollet outlines under Objectness priors: Solving this task requires
    object cohesion which he defines as the "Ability to parse grids into 
    “objects” based on continuity criteria including color continuity 
    or spatial contiguity"
    
    This solution uses a Scikit-learn linear regression model to predict
    the solution's final shape based on the training examples.
    
    This function first finds the unique colours in the intial grid and then
    for each colour removes the surrounding rows and columns (the solution 
    actually rotates the shape rather than deleting the columns) by deleting 
    the ones that don't contain that colour. This leaves two rectangular 
    shapes. Both remaining shapes are then padded with different combinations
    of filler to match the final shape. 
    
    Shapes Before filler:
        
    [[2, 2]
     [2, 0]
     [2, 2]]
    
    [[0, 4]
     [4, 4]
     [0, 4]]

    Possible filler combinations (this example only has horizontal filler):
        
    [[2, 2, 0]   [[0, 2, 2]
     [2, 0, 0]    [0, 2, 0]
     [2, 2, 0]]   [0, 2, 2]]
    
    [[0, 0, 4]   [[0, 4, 0]
     [0, 4, 4]    [4, 4, 0]
     [0, 0, 4]]   [0, 4, 0]]
    
    Bad combinations will create new colours or having black spaces remaing eg:
        
    [[2, 6, 0]
     [6, 4, 0]
     [2, 6, 0]]
    
    The combinations of shapes are then merged (added) together till a 
    solution that only contains the original two colours is found. 

    Parameters:
    x (np.array): Grid to solve

    Returns:
    np.array: Grid with result
      
    """
    
    # Get the predicted size of the result using the size of the shape
    target_h = predict_size('681b3aeb', x, size_of_shape)
    target_w = target_h

    # Get unique colours from grid excluding black 
    colours = np.delete(np.unique(x), [0])
    shapes = []
    
    # For each unique colour found
    for colour in colours:
        # Delete any rows that don't contain the colour
        shape = del_squares(x, colour)
        # Rotate and delete any row that don't contain the colour
        # This should leave a rectangle containing mostly just the shpae
        shape = del_squares(np.rot90(shape), colour)
        # Return the shape to original orientation
        shape = np.rot90(shape,3)
        # Add to list of found shapes
        shapes.append(shape)
     
    candidates = []    
    # For each shape found
    for shape in shapes:
        # Get the width and height
        h, w = shape.shape

        # Create the filler needed to create the target shape
        if h < target_h:
            filler_h = np.full((target_h-h,w), 0)
        else:
            filler_h = None
        if w < target_w:
            filler_w = np.full((target_h,target_w-w,), 0)
        else:
            filler_w = None
        
        # If a shape is already in the target shape then add to candidate list as is
        if w == target_w and h == target_h:
             candidates.append(shape)

        
        # Get the permuations of filler and shape
        # so it can try had to top or bottom / left or right
        perm_h = list(itertools.permutations([filler_h, shape]))       
        perm_w = list(itertools.permutations([filler_w, shape]))
        
        # Create all the combinations of filler and shape
        combinations = itertools.product(perm_h, perm_w)

        # For each combination
        for h, w in combinations:
            h_a, h_b = h
            w_a, w_b = w
            
            # Combine filler and shape
            bool_h = (h_a is not None and h_b is not None)
            if bool_h:
                s = np.concatenate((h_a,h_b), axis=0)
            if w_a is not None and w_b is not None:
                if np.count_nonzero(w_a == 0) == w_a.size and bool_h:
                    w_b = s
                elif bool_h:
                    w_a = s
                s = np.concatenate((w_a,w_b), axis=1) 
            candidates.append(s)
   
    solution = np.empty(0)
        
    # Get a pair of each combination from the candidates     
    for shape_a, shape_b in list(itertools.permutations(candidates,2)):
        # Add the shapes together
        solution = np.add(shape_a,shape_b)
        # If the only colours in the final shape are the colours of the shapes 
        # with no black squares the solution is found and the loop is exited
        compare = np.unique(solution) == np.unique(colours)
        if isinstance(compare, bool):
            match = compare
        elif isinstance(compare, np.ndarray):
            match = compare.all()
        if match:
            break
    
    return solution


def solve_ac0a08a4(x):
    """
    # Solves the task ac0a08a4. For this task the intial grid contains a 
    3 x 3 grid with a variable number of coloured squares. The solution is
    a larger square which 3 x the number of coloured squares bigger then the 
    intial grid. the ratio and placement of the coloured squares to black 
    squares remains the same. Solving this requires Numbers and Counting priors
    as outlined by Chollet.

    This solution uses a Scikit-learn linear regression model to predict
    the solution's final shape based on the training examples.
    
    This function uses the convert_shapes which has been generalized to work
    with a number of the solutions. It takes the intial grid and resizes it
    to a new target size which keeps all the ratios the same. 
    
    For this solution it increase the result shapes size based on the predicted
    new size using the train set of data. 

    Parameters:
    x (np.array): Grid to solve

    Returns:
    np.array: Grid with result
      
    """
    
    # Get predicted size of result using the number of colours
    target_h = predict_size('ac0a08a4', x, number_of_colours)
    target_w = target_h
    
    return convert_shapes(x, target_h, target_w)

def solve_b91ae062(x):
    """
    # Solves the task b91ae062. For this task the intial grid contains a 
    a grid with a variable number of coloured squares. The solution is
    a smaller square of size 3x3. the ratio and placement of the coloured squares to black 
    squares remains the same. THIS IS THE INVERSE OF ac0a08a4. The only difference
    in the code is the task id passed in to train the linear regression model. 
    This is to show adapability (albeit within a narrow skill and not even close
    to scale Chollet mentions in his paper for his meaures of an agent's ability to 
    achieve goals) which is allowed by using the training data to predict the 
    resulting grid size. 
    
    This function uses the convert_shapes which has been generalized to work
    with a number of the solutions. It takes the intial grid and resizes it
    to a new target size which keeps all the ratios the same. 
    
    For this solution it increase the result shapes size based on the predicted
    new size using the train set of data. 

    Parameters:
    x (np.array): Grid to solve

    Returns:
    np.array: Grid with result
      
    """
    
    # Get predicted size of result using the number of colours
    target_h = predict_size('b91ae062', x, number_of_colours)
    target_w = target_h
    
    return convert_shapes(x, target_h, target_w)

def solve_5ad4f10b(x):
    """
    # Solves the task 5ad4f10b. For this task the intial grid contains a 
    a grid with a variable number of coloured squares surrounded by black squares
    and noise (another colour randomly placed around). The solution is a 3 x 3
    counting the same geomtric shape of the coloured squares minus the noise.
    
    Solving this combines the Objectness priors and Number and counting priors
    of the other solutions and combines similar funcitonality from them.
    
    This solution first finds the which colour is clustered closet together
    by finding the distance between the first and last index of each colour. 
    The colour clustered closest together is most likely our squares (using 
    the most common colour does not work for all solutions). The rest of the 
    colours are then converted to black and any rows not contain the target colour 
    are strip away. Finally it then uses the convert_shapes function to get 
    the resulting smaller grid while keep the same geometric pattern. 

    Parameters:
    x (np.array): Grid to solve

    Returns:
    np.array: Grid with result
      
    """
    
    # Get predicted size of result using the size of the train shapes
    target_h = predict_size('5ad4f10b', x, size_of_shape)
    target_w = target_h

    target_colour, change_colour = 0,0
    # Get unique colours from grid excluding black 
    colours = np.delete(np.unique(x), [0])
   
    # Find the colour with the least distance between first last index
    max_dist= 0
    for colour in colours:      
        indexes = np.array(np.where(x == colour))
        x2,y2 = indexes[::,-1]
        x1,y1 = indexes[::,0]

        dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

        if dist < max_dist or max_dist == 0:
            target_colour = colour
            max_dist = dist
    
    # Get the shape by removing rows that don't contain the main colour
    for i in range(0,4):
        x = del_squares(np.rot90(x), target_colour)
        
    # Remove additional colours other than black and the main colour
    for colour in colours:
        if colour != target_colour:
            change_colour = colour
            x =  np.where(x == colour,0, x)
    
    x = np.where(x == target_colour, change_colour, x)
 
    return convert_shapes(x, target_h, target_w)

def predict_size(task, x, f):
    """
    # This function gets the data need to train the linear regression model
    to predict the result grid size

    Parameters:
    task (string): Task id to look up training data only
    x (np.array):  Grid to predict
    f (function):  Function for values to uses in prediction

    Returns:
    int: Predicted grid size
      
    """

    # for each task, read the data and call test()
    directory = os.path.join("..", "data", "training")
    json_filename = os.path.join(directory, task + ".json")
    data = read_ARC_JSON(json_filename)
    train_input, train_output, test_input, test_output = data
    
    #en(np.unique(item) ) - 1
    
    w = np.array([f(item) for item in train_input]).reshape(-1,1)
    y = np.array([item.shape[0] for item in train_output]).reshape(-1,1)

    z = np.array(f(x)).reshape(-1,1)
    
    result = train(w, y, z)

    return int(round(result[0][0]))
    

def train(w, y, z):
    """
    # This function uses the passed data to train and predict the size of the 
    result grid

    Parameters:
    w (np.array):  Train data 
    y (np.array):  Train data target result
    z (np.array):  Data to predict results for

    Returns:
    np.array: Array of predicted results
    """
      
    model = LinearRegression().fit(w, y)
    size = model.predict(z)
    return size

def size_of_shape(x):
    """
    # This function returns one dimension size of shpae

    Parameters:
    x (np.array):  Grid to get size from 

    Returns:
    int: One dimension size of grid passed in 
    """
    return x.shape[0]

def number_of_colours(x):
    """
    # This function returns number of colours in grid exluding black

    Parameters:
    x (np.array):  Grid to get size from 

    Returns:
    int: Unique number of colours not including black
    """
    return len(np.delete(np.unique(x) , [0]))

def del_squares(x, colour):
    """
    # This function returns a grid with the rows not contain the traget colour
    passed in removed

    Parameters:
    x (np.array):  Grid to get size from 
    colour (int):  Target colour

    Returns:
    np.array: Grid with only the rows containing the target colour
    """
    for i in reversed(range(0, len(x))):
        if colour not in x[i]:
            x = np.delete(x,i, axis=0)
            
    return x

def co_oridinates(l):
    """
    # Returns a list of co-ordinates for slicing based on the value l. Used for resizing 
    grids which contain 3 x 3 geomtry or 9 square shapes within the grid in total

    Parameters:
    l (int):  Size of each square in grid to be re-sized

    Returns:
    list: List of indexes for the starting point of each square with in the shape
    """
    # Get the step size for each square ie the length of each square shape within the grid 
    step =int(l/3)
    lst = []
    # For each square loop through and get the starting row and col index for that square
    for i in range(0, l, step):
        lst.append(list([i, i+step]))
    
    # Get all combinations of lst with itself for all row and col slice combinations (9 in total)
    return list(itertools.product(lst, lst))

def convert_shapes(x, target_h, target_w):
    """
    # Returns a re-sized grid. This works for grids that contain 9 equally sized
    shapes. co_oridinates functions returns the index for slicing of each new and old square 
    and then the new square is converted to the colour of the corresponding square from
    the original grid. 

    Parameters:
    x (np.array):   Original grid
    target_h (int): Target height of new shape
    target_w (int): Target width of new shape

    Returns:
    np.array: New re-sized grid
    """
    
    # Get the size of the orignal shape
    h, w = x.shape
    
    # Create new shape full of black squares for based on the target size 
    new_shape = np.full((target_h, target_w), 0)
    
    # Get co-oridinates for the index of each square in the original grid
    orig_co = co_oridinates(h)
    # Get co-oridinates for the index of each squre in the new gird
    new_co = co_oridinates(target_h)

    # For each of the 9 squares map the colour from old to new
    for i in range(0,9):
        row_o, col_o = orig_co[i]
        row_n, col_n = new_co[i]

        # Get the unique colour from the original sqaure
        uni = np.unique(x[row_o[0]:row_o[1], col_o[0]:col_o[1]])
        # Set colour of the new square to that of the original
        new_shape[row_n[0]:row_n[1], col_n[0]:col_n[1]] = [[uni[0]]]
                
    return new_shape

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

