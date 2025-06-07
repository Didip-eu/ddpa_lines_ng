#!/usr/bin/env python3

import itertools

from typing import Union,Tuple,List,Dict 



def flatten(l:Union[list,tuple], max_depth:int=-1):
    """ Flatten a sequence, Lisp-style. Input sequence may 
    be made of lists or tuples or both; the output sequence
    is always a list of lists.

    Args:
        depth (int): depth. Default (-1) flattens sublists at every level; 
            0 flattens only the top-level sublists, etc.
    Returns:
        list: a list.
    """
    def flatten_rec( l:list, dpt:int):
        if l == [] or l==() or (max_depth != -1 and dpt > max_depth):
            return []
        if type(l[0]) is not list and type(l[0]) is not tuple:
            return [l[0]] + flatten_rec(l[1:], dpt)
        return flatten_rec(l[0], dpt+1) + flatten_rec(l[1:], dpt)

    return flatten_rec( l, 0 )


def group(l:list, gs=2):
    """
    From a list, construct with same elements, but grouped in <gs>-size clusters.
    The last group may contain fewer elements.
    """
    groups = []
    for i,elt in enumerate(l):
        if i%2 == 0:
            groups.append([elt])
        else:
            groups[-1].append(elt)
    return groups




def deep_sorted(list_of_lists: List[Union[str,list]]) ->List[Union[str,list]]:
    """Sort a list that contains either lists of strings, or plain strings.
    Eg.::

       >>> _deep_sorted(['a', ['B', 'b'], 'c', 'd', ['e', 'E'], 'f'])
       [['B', 'b'], ['E', 'e'], 'a', 'c', 'd', 'f']

    Args:
        list_of_lists (List[Union[str,list]]): a list where each element can be a characters or a
            list of characters.

    Returns:
        List[Union[str,list]]: a sorted list, where each sublist is sorted and the top sorting
            key is the either the character or the first element of the list to be sorted.
    """
    return sorted([sorted(i) if len(i)>1 else i for i in list_of_lists],
                   key=lambda x: x[0])

def merge_sublists( symbol_list: List[Union[str,list]], merge:List[list]=[] ) -> List[Union[str,list]]:
    """Given a nested list and a list of strings, merge the lists contained in <symbol_list>
    such that characters joined in a <merge> string are stored in the same list.

    Args:
        merge (List[list]): for each of the provided subsequences, merge those output sublists
            that contain the characters in it. Eg. ``merge=['ij']`` will merge the ``'i'``
            sublist (``['i','I','î',...]``) with the ``'j'`` sublist (``['j','J',...]``)

    Returns:
        List[Union[str,list]]: a list of lists.
    """
    if not merge:
        return symbol_list

    symbol_list = symbol_list.copy()

    to_delete = []
    to_add = []
    for mgs in merge:
        merged = set()
        for charlist in symbol_list:
            if set(charlist).intersection( set(mgs) ):
                merged.update( charlist )
                to_delete.append( charlist )
        if len(merged):
            to_add.append( list(merged) )
    for deleted_subset in to_delete:
        try:
            symbol_list.remove( deleted_subset )
        except ValueError:
            print(f'Could not delete element {deleted_subset} from list of symbols.')
    if len(to_add):
        symbol_list.extend( to_add )
    return symbol_list


def groups_from_groups(list_of_lists: List[Union[list,str]], atoms: set = None, exclude=[]) -> List[Union[list,str]]:
    """Given a list of lists and a list of atoms, return a list of lists that only contains
    those atoms that are in the second list.
    NOT tested with compound symbols (=multichar atoms).

    Args:
        list_of_lists (List[Union[list,str]]): sets of atoms which determine the grouping.
        atoms (set): set of individual items.

    Returns:
        List[Union[List,str]]: a list of individual atoms or list of atoms.

    Example::

        >>> groups_from_groups( ['1','2','3','9',['J','Ĵ'],['j','ĵ','ɉ'],'Q',['U','Ù','Ú','Ų'],'u','ù','ú','ũ'], ['u','2','%','9','j','Ų','J','Q','U','ũ'])
        ['2', '9', 'J', 'j', 'Q', ['Ų', 'U'], ['u', 'ũ'], '%']


    """
    if atoms is None:
        return [ list(l) if len(l)>1 else l for l in list_of_lists ]

    all_atom_set = ''.join( flatten( list_of_lists ))
    unknown_atoms = set( c for c in atoms if c not in all_atom_set )

    atoms = atoms.difference( unknown_atoms )
    keys=[ [ c in sl for sl in list_of_lists ].index(True) for c in atoms]
    list_of_lists_new = []
    keyfunc = lambda x: x[0]
    list_of_lists_new = [ [ t[1] for t in l ] for k, l in itertools.groupby( sorted( zip(keys, atoms), key=keyfunc), key=keyfunc) ]

    return [ l[0] if len(l)==1 else l for l in list_of_lists_new ] + list(unknown_atoms)

def unzip(seq: Union[List[tuple],Tuple[tuple]] )->List[tuple]:
    """ Unzip a list of sequences. Eg.

    ```
    >>> unzip( [(1, 3, 5), (2, 4, 6))] )
    [(1, 2), (3, 4), (5, 6)]

    ```
    """
    return list(zip(*seq))

