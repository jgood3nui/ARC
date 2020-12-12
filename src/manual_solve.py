#!/usr/bin/python
# https://github.com/jgood3nui/ARC.git
# Student Name: Jonathan Good
# Student Number: 20235990

import os, sys
import json
import numpy as np
import re
import itertools

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_681b3aeb(x):

    target_h,target_w = 3,3

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

def solve_b2862040(x):
    return x

def solve_05269061(x):
    return x


def del_squares(x, colour):
    for i in reversed(range(0, len(x))):
        if colour not in x[i]:
            x = np.delete(x,i, axis=0)
            
    return x

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

