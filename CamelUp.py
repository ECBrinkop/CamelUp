# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:07:12 2021

@author: eikeb
"""
import copy, time
from random import randrange
import numpy as np, threading as th, pandas as pd
import numba as nb
from utils import print_header as print_hint2, print_adj
import itertools
import time
import random as rd
import sys

class CamelUp():
    '''
    Main Class for Camel Cup Game Analytics.
    This class can run a game and calculate expected payoffs for possible actions.
    ———————————————————————————————————————————————————————————————
    ———————————————————————————————————————————————————————————————
    Parameters
    n_players: int
        Number of players to join the game
    start: bool
        
    
    !!! simulation mode should be implemented to test certain strategies.
    
    
    Class lists include:
        Camels: 
            5 Camels (Purple, Blue, Orange, Green, Yellow)
        Inventory:
            Betting Plates for each of the Camels
    '''
    gap_margin = 5
    standard_Camels = ["Purple", "Blue", "Orange", "Yellow", "Green"]
    standard_Inventory = [] # contains 
    for i in standard_Camels:
        standard_Inventory.extend([i+" [5]",i+" [3]",i+" [2]"])
    field_structure = [[14,15,0,1,2],
                       [13,None,None,None,3],
                       [12,None,None,None,4],
                       [11,None,None,None,5],
                       [10,9,8,7,6]]
    center_design_standard = [
        '        <16>     < 1>     < 2>        |',
        '   <15>                        < 3>   |',
        '                                      |',
        ' <14>                            < 4> |',
        '                                      |',
        '                                      —',
        '                                      |',
        '         Value of Information:        |',
        ' <13>            {VOI:5.3f}           < 5> |',
        '                                      |',
        '                                      |',
        '                                      —',
        '                                      |',
        ' <12>                            < 6> |',
        '                                      |',
        '   <11>                        < 7>   |',
        '        <10>     < 9>     < 8>        |',
        '———————————————————————————————————————']
    center_design_extended = [
        '                                                                    ║',
        '             <16>               < 1>               < 2>             ║',
        '     <15>                                                  < 3>     ║',
        '                                                                    ║',
        '                                                                    ║',
        '   <14>                                                      < 4>   ║',
        '                                                                    ║',
        '                                                                    ═',
        '                                                                    ║',
        '                                                                    ║',
        '                ═══════════════════════════════════                 ║',
        '   <13>         ║ Value of Information:     {VOI:5.3f} ║          < 5>   ║',
        '                ═══════════════════════════════════                 ║',
        '                                                                    ║',
        '                                                                    ║',
        '                                                                    ═',
        '                                                                    ║',
        '   <12>                                                      < 6>   ║',
        '                                                                    ║',
        '                                                                    ║',
        '     <11>                                                  < 7>     ║',
        '             <10>               < 9>               < 8>             ║',
        '                                                                    ║',
        '═════════════════════════════════════════════════════════════════════']
    print_dim_standard = [31,76]
    print_dim_extended = [41,126]
    mask = {"Purple":1,"Blue":2,"Orange":3,"Yellow":4,"Green":5, "White":6, "Black":7}
    render_field_cell_width_standard = 12
    render_field_cell_width_extended = 22
    def __init__(self,
                 n_players:int,
                 start:bool = True,
                 field = "",
                 user_guide=True,
                 black_white= False): ## DONE!
        self.black_white = black_white
        self.Camels = copy.deepcopy(self.standard_Camels)
        self.Inventory = copy.deepcopy(self.standard_Inventory)
        self.render_field_cell_width = self.render_field_cell_width_standard
        self.print_dim = self.print_dim_standard
        self.center_design = self.center_design_standard

        ## settings if extended game is played
        if self.black_white:
            self.Camels.append("Black")
            self.Camels.append("White")
            self.Inventory.extend([f"{i} [2]" for i in self.standard_Camels])
            self.render_field_cell_width = self.render_field_cell_width_extended
            self.print_dim = self.print_dim_extended
            self.center_design = self.center_design_extended

        ## freeze self.Camels to tuple to avoid changes in the list
        self.Camels = tuple(self.Camels)

        self.total_width = (5*self.render_field_cell_width+6)
        self.user_guide=user_guide
        self.timers = []
        self.fields_payoffs = {}
        self.game_winner = []
        self.game_loser = []
        self.moved = []
        self.VOI = 0
        self.base_probabilities = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
        self.base_payoffs = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
        self.game_inventory = copy.deepcopy(self.Inventory)
        self.game_field = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        self.players = {}
        if field != "": 
            self.players = field.players
            self.game_field = field.game_field
            self.moved = field.moved
        else:
            for i in range(n_players):
                print("\nName of Player ["+str(i+1)+"]:")
                while True:
                    name = input()
                    if name == "":
                        name = "Player ["+str(i+1)+"]"
                    if name in self.players.keys() or name in self.Camels:
                        print("Invalid name (already existant or name of Camel), retry:")
                    else:
                        self.players[name] = player(name)
                        break
            self.position(start)
        print("—"*self.total_width)
        self.print_game(True, False)
        self.print_c()
        print("—"*self.total_width)

    def position(self,start=False): ## DONE!
        """
        This function is used to position the Camels on the field if the field is empty.
        """
        print("\n######################\nPositions in the game:\n######################")
        self.print_game()
        for i in self.Camels:
            print("\nPosition of Camel "+i+"?")
            pos = int(input())-1
            if len(self.game_field[pos]) == 0:
                self.game_field[pos].append(i)
            else:
                print("\nField not empty!")
                numbers = [len(j) for j in self.game_field[pos]]
                print_numbers = "'0'"
                print_field   = "   "
                for j in range(len(numbers)):
                    print_numbers += " "*(numbers[j]+2)+"'"+str(j+1)+"'"
                    print_field += " "+self.game_field[pos][j]+"    "
                print(print_field)
                print(print_numbers)
                print("\nWhere on the field?")
                pos_1 = int(input())
                self.game_field[pos] =[*self.game_field[pos][:pos_1],i,*self.game_field[pos][pos_1:]]
        if start:
            for player in self.players.keys():
                self.OasisDesert(player)
        
    def OasisDesert(self,player,plate = ""): ## DONE!
        """
        This function is used to set the Oasis/Desert on the field for a player.
        """
        if player not in self.players.keys():
            print("\n!!!!!!!!!!!!!!!!!!!!\nInvalid player name!\n!!!!!!!!!!!!!!!!!!!!\n")
            return 0
        for i in range(len(self.game_field)):
            if player in self.game_field[i]:
                self.game_field[i] = []
                self.players[player].plate_pos = None
                break
        plate = plate.capitalize()
        while True:
            if plate == "":
                print(player+"'s Oasis/Desert on the field?\n 'o' for OASIS, 'd' for DESERT, "+\
                    "'w' to WITHDRAW plate")
                plate = input().capitalize()
            if plate in ["O", "OASIS"]:
                plate = "OASIS"
                break
            elif plate in ["D", "DESERT"]:
                plate = "DESERT"
                break
            elif plate in ["W", "WITHDRAW"]:
                plate = "WITHDRAW"
                break
            else:
                print("Invalid input, retry:")
        
        while True:
            if plate == "WITHDRAW":
                break
            print("On which position is the "+plate+"?")
            # self.print_game()
            pos = int(input())-1
            if len(self.game_field[pos]) != 0 or pos in [0,16,17,18]:
                print("Invalid field, retry:")
            elif "OASIS" in self.game_field[pos-1] or "OASIS" in self.game_field[pos+1] or\
                "DESERT" in self.game_field[pos-1] or "DESERT" in self.game_field[pos+1]:
                    print("Invalid field, retry!")
            else:
                self.game_field[pos] = [plate,player]
                if str(pos+1)+"O" in self.fields.keys():
                    if plate == "OASIS":
                        self.game_field[pos]+=[self.fields[str(pos+1)+"O"]]
                    else:
                        self.game_field[pos]+=[self.fields[str(pos+1)+"D"]]
                self.players[player].plate_pos = pos
                break
        # self.print_game()
    def print_game(self,
                   field=True,
                   payoffs=True,
                   ): ## DONE!
        """
        This function is used to print the game field and payoffs. This depends on the settings of the game.
        """
        print(self.print_dim)
        self.rendered_output = [""]*self.print_dim[0]
        self.rendered_header = [""]*3
        if field:
            self.print_render_field()
        if payoffs:
            if self.base_probabilities.sum() == 0:
                self.one_turn(print_option=False,OD=False,player = list(self.players.keys())[0])
            self.print_render_payoffs()
        print("\n"+"\n".join(self.rendered_header)+"\n") ## prints header that was filled in print_render_field()
        print("\n".join(self.rendered_output)) ## prints the field that was filled in print_render_field()
        
    def print_render_field(self): ## DONE! pending testing
        """
        This function is used to render the game field for printing.
        Uses self.game_field and self.field_structure 
         and the natural shape of the CamelUp to render 
         the field output into a 31x66 pixel(well, 31x66 characters) image if standard game is played
         or a 41x116 pixel(41x116 characters) image if extended game is played.
        """
        total_width = self.total_width
        gap_margin = self.gap_margin ## margin to left side for printing
        
        ## add margins to left side for printing for the field and the header
        for row in range(len(self.rendered_output)):
            self.rendered_output[row]+=" "*gap_margin
        for row in range(len(self.rendered_header)):
            self.rendered_header[row]+=" "*gap_margin
        
        ## header rendering
        header_statement = "Current Field" ## header statement for the field
        self.rendered_header[0] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = total_width-18) ## central header directly above field for all 3 rows
        self.rendered_header[1] += "{string:^{width}s}".format(\
                                    string="##  "+header_statement+"  ##",
                                    width = total_width-18)
        self.rendered_header[2] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = total_width-18)

        ## field rendering
        vertical_sign = "║" if self.black_white else "|"
        horizontal_sign = "═" if self.black_white else "—"
        self.rendered_output[0]+= horizontal_sign*total_width
        local_center_design = self.center_design+[]
        ## format VOI into field center depending on game mode
        if self.black_white:
            local_center_design[8] = local_center_design[8].format(VOI=self.VOI)
        else:
            local_center_design[11] = local_center_design[11].format(VOI=self.VOI)
        n_rows_cells = 5 if not self.black_white else 7 ## number of rows in a cell in the field depending on game mode
        ## iterate over printing field cells
        for row_n in range(5):
            for column_n in range(5):
                if row_n in [1,2,3] and column_n in [1,2,3]:
                    if column_n ==1:
                        for render_row in range(row_n*(n_rows_cells+1),row_n*(n_rows_cells+1)+n_rows_cells+1):
                            if "VOI" in local_center_design[render_row-(n_rows_cells+1)]:
                                self.rendered_output[render_row+1] += local_center_design[render_row-(n_rows_cells+1)].format(VOI=self.VOI)
                                continue
                            self.rendered_output[render_row+1]+=local_center_design[render_row-(n_rows_cells+1)]
                    continue
                field_n = self.field_structure[row_n][column_n] ## get field number
                field_n_content = copy.deepcopy(self.game_field[field_n])
                field_contents = []
                if "O"+str(field_n) in self.fields_payoffs.keys() or\
                    "D"+str(field_n) in self.fields_payoffs.keys():
                    if field_n_content == []:
                        field_contents = ["(DESERT)","({string:.2f})".format(string=self.fields_payoffs["D"+str(field_n)]),
                                            "(OASIS)","({string:.2f})".format(string=self.fields_payoffs["O"+str(field_n)])]
                    elif field_n_content[0] in ["DESERT","OASIS"]:
                        #field_n_content +=[np.nan,"",""]
                        if field_n_content[0] == "OASIS":
                            field_contents = ["OASIS","({string:.2f})".\
                                format(string=self.fields_payoffs["O"+str(field_n)]),
                                "(DESERT)","({string:.2f})".format(string=self.fields_payoffs["D"+str(field_n)])]
                        else:
                            field_contents = ["DESERT","({string:.2f})".\
                                format(string=self.fields_payoffs["D"+str(field_n)]),
                                "(OASIS)","({string:.2f})".format(string=self.fields_payoffs["O"+str(field_n)])]
                        if "W"+str(field_n) in self.fields_payoffs.keys():
                            field_contents.insert(0,"(({string:.2f}))".\
                                format(string=self.fields_payoffs["W"+str(field_n)])) ## TODO: check if this is correct
                    for i in range(len(field_contents)):
                        field_contents[i] = f"{field_contents[i]:^{self.render_field_cell_width}s}{vertical_sign}"
                ## contents are rendereded for each cell row
                if field_n_content == []:
                    pass
                elif field_n_content[0] in ["DESERT","OASIS", "(DESERT)"]:
                    if len(field_contents) == 0:
                        for i in range(len(field_n_content)):
                            field_contents.append(f"{field_n_content[i]:^{self.render_field_cell_width}s}{vertical_sign}")
                    #else:
                    #    for i in range(len(field_n_content)):
                    #        field_contents[i] = f"{field_contents[i]:^{self.render_field_cell_width}s}{vertical_sign}"
                else:
                    field_n_content = field_n_content[::-1]
                    for row_o in range(len(field_n_content)):
                        camel = field_n_content[row_o]
                        if not self.black_white and camel == "Purple":
                            camel = "White"
                        if camel in self.moved or camel == "Black" and "White" in self.moved:
                            camel = f"[{camel}]"
                        field_contents.append(f"{camel:^{self.render_field_cell_width}s}{vertical_sign}")

                while len(field_contents) < n_rows_cells: ##elongate the cell for the content to fit in the cell
                    field_contents.append(" "*self.render_field_cell_width+vertical_sign)
                    if len(field_contents) < n_rows_cells:
                        field_contents.insert(0," "*self.render_field_cell_width+vertical_sign)
                field_contents = field_contents + [horizontal_sign*(self.render_field_cell_width+1)]
                for row_m in range(n_rows_cells+1):
                    render_row = row_m+row_n*(n_rows_cells+1)+1
                    if column_n == 0:
                        extra_sign = vertical_sign
                        if row_m == n_rows_cells:
                            extra_sign = horizontal_sign
                        self.rendered_output[render_row]+= extra_sign
                    try:
                        self.rendered_output[render_row]+= field_contents[row_m]
                    except:
                        print(
                            "render_row:",render_row,"row_m:",row_m,
                            "field_contents[row_m]:",field_contents[row_m],
                            "self.rendered_output[render_row]:",self.rendered_output[render_row])
                        raise Exception("Error in print_render_field")

            
    def print_render_payoffs(self): ## Done! untested!!!
        '''
        Renders the payoffs for printing purposes
        
        The payouts are layouted to the right side of the field.
        The elements are:
            - Win Probabilities 4x6 or 6x4
            - Expected Payoffs of plates. 6x5 or 5x6
            - Coins and expected Payoffs of players up to 8x3 or 3x8.
            - Camels not diced 1x7.
        the space that this function has is either 41 or 31 rows, depending on the game mode. 
            ## 2 space between lines, 1 after odd lines, 4-3 space if extended game is played.
            extended game: 41 rows 5, (win probs) 4, 6, (expected payoffs) 5, 6, (coins) 3, 6, (camels not diced) 1, 5.
            standard game: 31 rows 3, (win probs) 4, 4, (expected payoffs) 5, 4, (coins) 3, 4, (camels not diced) 1, 3.
            - Player Inventories and expected payoffs of items in inventory. Structured list below the field.
        '''
        gap_margin = self.gap_margin
        
        i = 0 if not self.black_white else 2
        margins = np.array([3,4,4,4,3])+i
        ## if more than 4 players, lower the margins in the middle so two rows of players fit.
        if len(self.players) >4: 
            margins[1:4] -= 1
        margins = margins.tolist()
        ## !!! marker 

        width = self.print_dim[1]-self.total_width
        header_statement = "Win Probabilities"
        self.rendered_header[0] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = width)
        self.rendered_header[1] += "{string:^{width}s}".format(\
                                    string="##  "+header_statement+"  ##",
                                    width = width)
        self.rendered_header[2] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = width)

        row_n = margins[0]

        probabilities = pd.DataFrame(self.base_probabilities, index=self.Camels[:5], columns=["First","Second","Lose"])
        payoffs = pd.DataFrame(self.base_payoffs, index=self.Camels[:5], columns=["5-Plate","3-Plate","2-Plate"])
        prob_strings, payoffs_strings = format_tables(probabilities, payoffs, self.game_inventory, self.black_white)
        for row in range(len(prob_strings)):
            self.rendered_output[row_n] += " "*gap_margin + prob_strings[row]
            row_n += 1
        row_n += margins[1]
        for row in range(len(payoffs_strings)):
            self.rendered_output[row_n] += " "*gap_margin + payoffs_strings[row]
            row_n += 1
        row_n += margins[2]
        
        ### header of player coins
        row_n = self.print_c(row_n)

        row_n += margins[3]
        self.rendered_output[row_n] += " "*gap_margin + "Camels not diced: " + " ".join([i for i in self.Camels if i not in self.moved])
        row_n += margins[4]

        self.rendered_output.extend([""," "*gap_margin + "Player Inventories and expected payoffs:"])

        ### print player inventories and expected payoffs
        for player in self.players.values():
            self.rendered_output.extend([""," "*gap_margin + "## " +player.name+" ##"])
            self.rendered_output.append(" "*gap_margin + "Inventory: ")
            dice_counts = 0
            if "Diced" in player.inventory:
                dice_counts = player.inventory.count("Diced")
            self.rendered_output[-1] += f"Diced: {dice_counts}, "
            if player.plate_pos != None:
                self.rendered_output[-1] += f"Plate[{player.plate_pos+1}]: {player.plate_value}, "
            for element, payoff in zip(player.inventory, player.inventory_payoffs):
                if element == "Diced":
                    continue
                else:
                    self.rendered_output[-1] += f"{element}: {payoff}, "

    def print_c(self, index = 0): ## DONE! 
        if index == 0:
            print_hint2("Coins of players:")
        gap_margin = self.gap_margin
        range_par = min(len(self.players),4)
        if index == 0:
            row_n = 0
            self.rendered_output = []
        else:
            row_n = index
        self.rendered_output += [""]*3
        if len(self.players) > 4 and index == 0:
            self.rendered_output += [""]*3
        self.rendered_output[row_n] += " "*gap_margin + "Players   "
        lengths = [max(len(player.name)+2,7) for player in self.players.values()]
        names = [player.name for player in self.players.values()]
        self.rendered_output[row_n] += "".join([f"{names[i]:<{lengths[i]}} " for i in range(range_par)])
        ### player current coins
        row_n += 1
        self.rendered_output[row_n] += " "*gap_margin + "Coins     "
        self.rendered_output[row_n] += "".join([f"{self.players[names[i]].coins:<{lengths[i]}} " for i in range(range_par)])
        ### player current expected payoff
        row_n += 1
        self.rendered_output[row_n] += " "*gap_margin + "EV        "
        self.rendered_output[row_n] += "".join([f"{self.players[names[i]].expected_payoff:<{lengths[i]}} " for i in range(range_par)])
        if len(self.players) > 4:
            ### header of player coins
            row_n += 1
            self.rendered_output[row_n] += " "*gap_margin + "Players   "
            lengths = [max(len(player.name)+2,7) for player in self.players.values()[4:]]
            self.rendered_output[row_n] += "".join([f"{names[4+i]:<{lengths[i]}} " for i in range(len(self.players.values()[4:]))])
            ### player current coins
            row_n += 1
            self.rendered_output[row_n] += " "*gap_margin + "Coins     "
            self.rendered_output[row_n] += "".join([f"{self.players[names[4+i]].coins:<{lengths[i]}} " for i in range(len(self.players.values()[4:]))])
            ### player current expected payoff
            row_n += 1
            self.rendered_output[row_n] += " "*gap_margin + "EV        "
            self.rendered_output[row_n] += "".join([f"{self.players[names[4+i]].expected_payoff:<{lengths[i]}} " for i in range(len(self.players.values()[4:]))])
        if index == 0:
            print("\n".join(self.rendered_output))
        else:
            return row_n

    def moved_f(self,camel):  ## Done!
        if camel not in self.moved and camel in self.Camels:
            self.moved.append(camel)
            if camel == "Black":
                self.moved.append("White")
            elif camel == "White":
                self.moved.append("Black")

    def cl(self): ## DONE! 
        """
        This function clears the game and prepares for a new round.
        """
        self.one_turn(False)
        self.moved = []
        for i in range(16):
            if "OASIS" in self.game_field[i] or \
                "DESERT" in self.game_field[i]:
                self.game_field[i]=[]
        self.game_inventory = copy.deepcopy(self.Inventory)
        for i in self.players.keys():
            self.players[i].end_of_round()
        self.print_c()

    def move(self,camel,steps): ## Done!
        if camel not in self.Camels:
            print("Invalid Camel!")
            return 0
        else:
            self.moved_f(camel)
        for j in range(len(self.game_field)):
            if camel in self.game_field[j]:
                index = self.game_field[j].index(camel)
                moving_camels = self.game_field[j][index:]
                self.game_field[j] = self.game_field[j][:index]
                if camel not in ["Black","White"]:
                    field = j + steps
                else:
                    field = j - steps
                break
        if "OASIS" in self.game_field[field]:
            player = self.game_field[field][1]
            print("#"*len(player)+"#############\n"+player+" gets a coin!\n"+"#"*len(player)+"#############\n")
            self.players[self.game_field[field][1]].coins+=1
            if camel not in ["Black","White"]:
                field += 1
            else:
                field -= 1
            self.game_field[field].extend(moving_camels)
        elif "DESERT" in self.game_field[field]:
            player = self.game_field[field][1]
            print("#"*len(player)+"#############\n"+player+" gets a coin!\n"+"#"*len(player)+"#############\n")
            self.players[self.game_field[field][1]].coins+=1
            if camel not in ["Black","White"]:
                field -= 1
            else:
                field += 1
            moving_camels.extend(self.game_field[field])
            self.game_field[field] = moving_camels
        else:
            self.game_field[field].extend(moving_camels)

    def make_game_inventory_matrix(self,game_inventory): ## Done!
        '''
        This function creates a matrix of the game inventory.
        '''
        self.game_inventory_matrix = np.zeros((5,3))
        for i in game_inventory:
            color, number = i.split(" ")
            number = [5,3,2].index(int(number[1]))
            color_index = self.Camels.index(color)
            self.game_inventory_matrix[color_index, number] += 1

    def render_camels_die(self, masked=True):
        """
        This function renders the camels that haven't moved.
        """
        # determine camels that haven't moved:
        Camels_die = list(self.Camels)
        for i in self.moved:
            if i in Camels_die:
                del Camels_die[Camels_die.index(i)]

        if not masked:
            return Camels_die, 6-len(Camels_die_rendered) if self.black_white else 5-len(Camels_die_rendered)
        ## encode camels that still die
        Camels_die_rendered = [self.mask[i] for i in Camels_die]
        ## handle special case of black and white camel
        if self.black_white:
            if 6 in Camels_die_rendered and 7 in Camels_die_rendered:
                Camels_die_rendered.remove(7)
            elif 6 not in Camels_die_rendered and 7 in Camels_die_rendered:
                Camels_die_rendered.remove(7)
            else:
                Camels_die_rendered.remove(6)
            n_camels_thrown = 6-len(Camels_die_rendered)
        else:
            n_camels_thrown = 5-len(Camels_die_rendered)
        return Camels_die_rendered, n_camels_thrown

    def one_turn(self,print_option=True,OD=False,player = ""): ## Done! untested!!!
        '''
        This function manages all the payoffs
        '''
        ## time one turn start: field rendering, game inventory matrix creation
        self.timers.append({"one_turn_start": time.time()})
        
        if player not in self.players.keys():
            print("Invalid player!")
            return
        
        Camels_die_rendered, n_camels_thrown = self.render_camels_die()

        # determine current field of play
        start_field = copy.deepcopy(self.game_field)
        start_players = copy.deepcopy(self.players)
        ## render said field
        rendered_field, player_mapping = render_field(start_field,start_players)
        
        ## render game inventory matrix
        self.make_game_inventory_matrix(self.game_inventory)
        self.timers[-1]["one_turn_start"] = time.time() - self.timers[-1]["one_turn_start"]

        ## time all_moves for base field
        self.timers.append({"sim_all_moves_start": time.time()})
        ## run simulation numba accelerated
        self.base_probabilities, self.base_DO_hits, VOI = sim_all_moves(
            rendered_field, len(self.players), n_camels_thrown, 
            Camels_die_rendered, self.game_inventory_matrix, verbose=False)
        self.timers[-1]["sim_all_moves_start"] = time.time() - self.timers[-1]["sim_all_moves_start"]
        payoff = {}
        
        for i in range(rendered_field.shape[0]):
            if rendered_field[i, 0]>7:
                player_index = rendered_field[i, 1]
                player_name = player_mapping[player_index] ## player position maps from player index + 10 to player name
                payoff[player_name] = {
                    "Type": "Desert" if rendered_field[i, 0] == 8 else "Oasis", 
                    "Field": i,
                    "Expected Payoff": self.base_DO_hits[player_index-10, 0]}

        self.base_payoffs = np.matmul(self.base_probabilities, np.array([[5,3,2],[1,1,1],[-1,-1,-1]]))
        #base_payoffs = pd.DataFrame(base_payoffs, index=self.Camels[:5], columns=["5-Plate", "3-Plate", "2-Plate"])

        ## handle desert value calculation for voi and differentials for desert plate changes (VOI is not needed for this)
        ## questions: How are player inventories handled? This is key to determining the expected payoffs of desert and oasis fields for the player at turn.
        ## Involves handling of DO plates, dice plates, and bet plates.
        ## ideally directly usable in field printing.
        
        ## create empty filter matrices for player inventories
        self.player_inventory_filter = [np.zeros((5,3))]*len(self.players)

        ## calculate expected payoffs for player inventories
        player_index = 0
        for i in self.players.keys():
            e_payoff = 0
            for j in self.players[i].inventory:
                if j == "Diced":
                    e_payoff+=1
                    self.players[i].inventory_payoffs.append(1)
                else:
                    color, number = j.split(" ")
                    number = int(number[1])
                    color_index = self.Camels.index(color)
                    number_index = [5,3,2].index(number)
                    EV_plate = self.base_payoffs[color_index, number_index]
                    self.player_inventory_filter[player_index][color_index, number_index] = 1
                    self.players[i].inventory_payoffs.append(EV_plate)
                    e_payoff += EV_plate
            if self.players[i].plate_pos != None:
                expeted_hits = self.base_DO_hits[player_index,0]
                e_payoff += expeted_hits
                self.players[i].plate_value = expeted_hits
            #print("e_payoff:",e_payoff)
            self.players[i].expected_payoff = round(e_payoff,2)
            player_index += 1
        
        self.fields_payoffs = {}
        player_index = list(self.players.keys()).index(player)
        ## Oasis and Desert value optimisation.
        if OD:
            self.timers.append({"desert_iterator": time.time()})
            ## calculate desert value for player for all legal fields:
            self.DO_fields_payoffs = self._desert_iterator(player) ## prepared for it does not have to be rerun every turn. !!! implement this!
            ## returns a dictionary of payoffs

            ## delta base is the expected payoff of the player's plate at the start of the turn.
            for i in self.DO_fields_payoffs.keys():
                field_payoff = self.DO_fields_payoffs[i]
                ## delta base is the expected payoff of the players plate minus the base payoff of the plate of the player.
                delta = -self.base_DO_hits[player_index,0] +field_payoff[1][player_index,0] ## create copy of delta base
                ## THIS player additionally gains, if the other players lose hits.
                delta_other_players_hits = np.delete(self.base_DO_hits, player_index)-np.delete(field_payoff[1], player_index)

                delta+=np.sum(delta_other_players_hits)

                ## Additionally, THIS player gains the deltas in his expected payoffs:
                current_payoffs = np.matmul(field_payoff[0],np.array([[5,3,2],[1,1,1],[-1,-1,-1]]))
                delta_payoff_matrix = current_payoffs-self.base_payoffs
                delta_inventory_payoffs = np.sum(delta_payoff_matrix*self.player_inventory_filter[player_index])
                delta += delta_inventory_payoffs

                ## Finally, the player gains the expected payoffs of other players bets with the old plate, 
                ## but loses the expected payoffs of other players bets with the new plate.
                for player_index_other in range(len(self.players)):
                    if player_index_other != player_index:
                        delta_other_players_bets = -np.sum(delta_payoff_matrix*self.player_inventory_filter[player_index_other])
                        delta += delta_other_players_bets

                self.fields_payoffs[i] = delta
            #print(self.fields_payoffs)
            self.timers[-1]["desert_iterator"] = time.time() - self.timers[-1]["desert_iterator"]
        ## game should be printed only after calculations for deserts and oases are complete and expected payoffs are calculated.
        if print_option:
            self.print_game(True,True)

    def _desert_iterator(self,player:str, verbose:int = 0): ## done, untested
        '''
        This function brute forces all potential desert and oasis fields for a given player. 
        The rule for taking these fields is: At max 4 fields in fron of any camel of the standard game. 
        At max 3 fields behind the white and black camels in the extended game.

        Parameters
        ----------
        player : str
            Player as reference, naturally must be in the players dictionary. 

        Returns
        -------
        fields: dict
            Potential fields to place desert or oasis on. Keys are 'W' for withdrawal, and then "D" or "O" plus a field index for 
            desert or oasis respectively. Values are a list of two arrays. The first array is the payoff matrix for the player's plate,
            the second array is the payoff matrix for the oasis or desert field.

        '''
        ## look for valuable fields for deserts
        field = self.game_field + []
        W_field = "xxx"

        start_players = copy.deepcopy(self.players)
        fields = {}
        ## withdrawal field:
        for i in range(len(field)):
            if player in field[i]:
                j = i#+1
                W_field = "W"+str(j)
                field[i] = []
                fields[W_field], _  = render_field(field, start_players)
                break

        ## calculate all possible fields for desert and oasis
        Camels = 0
        distance = 5
        Camel_distance = [5] * 16
        desert_found = False
        for i in range(1,16):
            distance +=1
            if set(field[i]) & set(self.Camels[:5]):
                Camel_distance[i] = 5
                distance = 0
                desert_found = False
            elif set(field[i]) & set(["White","Black"]):
                Camel_distance[i] = 5
            elif desert_found:
                Camel_distance[i] = 100
                desert_found = False
            elif field[i]==[]:
                Camel_distance[i] = distance
            elif set(field[i]) & set(["DESERT","OASIS"]):
                Camel_distance[i] = 5
                desert_found = True
        ## special case for extended game:
        distance = 5; desert_found = False;
        if self.black_white:
            for i in range(15,-1,-1):
                distance +=1
                if set(field[i]) & set(["Black","White"]):
                    distance = 1
                    desert_found = False
                elif desert_found:
                    Camel_distance[i] = 100
                    desert_found = False
                elif field[i] == [] and Camel_distance[i] != 100:
                    Camel_distance[i] = min(distance,Camel_distance[i])
                elif set(field[i]) & set(["DESERT","OASIS"]):
                    Camel_distance[i] = 5
                    desert_found = True

        legal_fields = [idx for idx,distance in enumerate(Camel_distance) if distance > 0 and distance < 5]

        for i in legal_fields:
            j = i#+1
            fields["D"+str(j)] = field.copy()
            fields["D"+str(j)][i] = ["DESERT",player]
            fields["D"+str(j)], _ = render_field(fields["D"+str(j)],start_players)
            fields["O"+str(j)] = field.copy()
            fields["O"+str(j)][i] = ["OASIS",player]
            fields["O"+str(j)], _ = render_field(fields["O"+str(j)],start_players)

        if verbose == 1:
            return fields.keys()
        ## camels diced

        Camels_die_rendered, n_camels_thrown = self.render_camels_die()

        fields_payoffs = {}
        for i in fields.keys():
            #print(type(fields[i]),type(len(start_players)),type(n_camels_thrown),type(Camels_die_rendered),type(self.game_inventory_matrix))
            fields_payoffs[i] = sim_all_moves(
                fields[i],len(start_players),n_camels_thrown,
                Camels_die_rendered,self.game_inventory_matrix, verbose = False)

        return fields_payoffs # dictionary of potential plate fields: returns and their payoff matrix

    def rank(self,field): ## DONE!
        ## returns ranks of camels in field
        ranks = []
        for i in field:
            if "OASIS" not in i and "DESERT" not in i and len(i) > 0:
                ranks.extend(i)
        if "Black" in ranks: ## special case for extended game:
            ranks.remove("Black")
            ranks.remove("White")
        return ranks

    def die_r(self): ## DONE!
        moving, _ = self.render_camels_die(False)
        camel = moving[randrange(len(moving))]
        if camel == "White":
            camel = rd.choice(["Black","White"])
        steps = randrange(1,4)
        lenstr = 6+7+7+1+len(camel)
        print("\n"+"#"*lenstr+"\nCamel "+camel+" moves "+str(steps)+" steps!\n"+"#"*lenstr+"\n")
        self.move(camel,steps)
    
    def game_not_end(self): ## DONE!
        if len(set([*self.game_field[16],*self.game_field[17],*self.game_field[18]]).difference(set(["Black","White"]))) == 0:
            return True
        else:
            return False

    def game(self,player = ""): ## DONE!
        '''
        Starts game with input as starting player.
        Manages moves by calling the make_a_move function
        Manages the end of a round by calling the cl function if all 5 dice are thrown.
        Manages the end of a game by calling the game_end function when a camel passes the line
        '''
        n_players = len(list(self.players.keys()))
        # starting player
        if player not in self.players.keys():
            while True:
                print_hint2("Start Player?")
                player = input()
                if player in self.players.keys():
                    break
                print("\nIllegal Player Name. Chose one of:\n\
                      \t{:s}\n".format(", ".join([player_i for player_i in self.players.keys()])))
        index = list(self.players.keys()).index(player)
        # actual game
        while self.game_not_end():
            player = list(self.players.keys())[index%n_players]
            self.make_a_move(player)
            if len(self.moved) == 5:
                self.cl() #end of round function
                if self.user_guide:
                    self.print_i()
            index += 1
        self.game_end() #end of game

    def game_end(self): ## DONE!
        self.moved = list(self.Camels)
        if "Black" in self.moved:
            self.moved.remove("Black")
        self.cl()
        self.print_c()
        ## determine loser:
        for i in self.game_field:
            if i != []:
                if i[0] not in ["OASIS","DESERT"]:
                    if i not in ["Black","White"]:
                        loser = i[0]
                        break
        ## determine winner:
        winner_stack = [*self.game_field[16],*self.game_field[17],*self.game_field[18]]
        for i in winner_stack[::-1]:
            if i not in ["Black","White"]:
                winner = i[0]
                break

        print_hint2(loser+" lost track and is dead last!")
        j = 0
        words = ["first","second","third","fourth","fifth","sixth"]
        coins = ["8","5","3","2","1","no"]
        for i in self.game_loser:
            if i[1] == loser and j < 5:
                print(i[0]+" predicted the loser correctly "+words[j]+" and gets "+coins[j],end = " ")
                if j == 0:
                    print("coins. Congratulations!")
                elif j != 4:
                    print("coins.")
                elif j == 4:
                    print("coin.")
                self.players[i[0]].coins+=int(coins[j])
                j+=1
            elif i[1] == loser and j >= 5:
                j+=1
                print(i[0]+f" predicted the loser correctly too late [{j}th] and loses a coin.")
                self.players[i[0]].coins-=1
            else:
                print(i[0]+" wrongly predicted the losing camel and loses a coin.")
                self.players[i[0]].coins-=1
        j=0
        print_hint2(winner+" won the race!!!!")
        for i in self.game_winner:
            if i[1] == winner and j < 5:
                print(i[0]+" predicted the winner correctly "+words[j]+" and gets "+coins[j],end = " ")
                if j == 0:
                    print("coins. Congratulations!")
                elif j != 4:
                    print("coins.")
                elif j == 4:
                    print("coin.")
                self.players[i[0]].coins+=int(coins[j])
                j+=1
            elif i[1] == winner and j >= 5:
                j+=1
                print(i[0]+f" predicted the winner correctly too late [{j}th] and loses a coin.")
                self.players[i[0]].coins-=1
            else:
                print(i[0]+" wrongly predicted the winning camel and loses a coin.")
                self.players[i[0]].coins-=1
        
        self.print_c()
        winner = []
        maxcoins = 0
        for i in self.players.keys():
            if self.players[i].coins>maxcoins:
                winner = [i]
                maxcoins = self.players[i].coins
            elif self.players[i].coins==maxcoins:
                winner.append(i)
        if len(winner) == 1:
            print_hint2("The winner iiiiiiiiis    "+winner[0]+"!!")
        else:
            print_win = ""
            for i in range(len(winner)-1):
                print_win+=", "+winner[i]
            print_win = print_win[2:]+" and "+winner[-1]+"!!"
            print_hint2("We have a TIE!! The winners aaaaaare "+print_win)

    def print_i(self):
        # print("PLATES AVAILABLE:\n")
        # print_available = ""
        # for j in [" [5]"," [3]"," [2]"]:
        #     for i in self.Camels:
        #         if i+j in self.game_inventory:
        #             print_available += print_adj(i+j, len(i+j))+" | "
        #         else:
        #             print_available += " "*len(i+j)+" | "
        #     print_available=print_available[:-3]+"\n"
        # print(print_available+"\n")
        print("CHOSE:\n")
        col0_len = 20
        print(print_adj("Action",10)+" | "+print_adj("Take top bet:",col0_len-5)+" | "+\
              print_adj("Throw dice:",col0_len-5)+" | "+print_adj("Throw dice random:",col0_len)+" | "+\
              print_adj("Set Oasis/Desert:",col0_len)+" | "+print_adj("Take final bet:",col0_len)+" | "+print_adj("Exit:",col0_len)+"\n"+\
                  "–"*(10+5*col0_len) )
        print(print_adj("Type",10,"c")+" | "+print_adj("[color]",col0_len-5,"c")+" | "+\
              print_adj("t",col0_len-5,"c")+" | "+ print_adj("r",col0_len,"c")+" | "+
              print_adj("o, d or w",col0_len,"c")+" | "+print_adj("f",col0_len,"c") + " | "+print_adj("q",col0_len,"c"))

    def make_a_move(self,player):
        # self.print_game(True,True)
        CNM = ""
        for i in self.Camels:
            if len(self.moved) == 0:
                CNM = "No Camel has moved yet!  "
                break
            if i not in self.moved:
                CNM+=print_adj(i,8,"l")+", "
        self.one_turn(print_option=False,OD=True,player=player)
        self.print_game(True, True) # prints game with field and payoffs and player gains
        #self.print_c()
        #print_hint2("Camels not moved: "+CNM[:-2])
        print_hint2(player+"'s move")
        if self.user_guide:
            self.print_i()
        sys.stdout.flush()
        while True:
            print("Move t for throw a camel, r for random throw, o or d or w for desert, oasis, [Camel] for bet, f for final bet")
            move = input()
            if move == "q":
                sys.exit()
            elif move == "t":
                print("Which camel?")
                move = input()
                if move.capitalize() in self.Camels:
                    if move.capitalize() not in self.moved:
                        print("How many steps did Camel "+move.capitalize()+" move?")
                        moves = input()
                        if moves not in ["1","2","3"]:
                            print("Illegal number of moves, retry!")
                        else:
                            print_hint2(player+" has diced and gets a coin plate!")
                            self.players[player].inventory.append("Diced")
                            self.move(move.capitalize(),int(moves))
                            break
                    else:
                        print("Camel "+move+" already moved! Retry!")
                        print(CNM)
                else:
                    print(move.capitalize()+" is not a camel! Retry!")
            elif move.capitalize() in self.Camels:
                plates = []
                for i in self.game_inventory:
                    if move.capitalize() in i:
                        plates.append(int(i[-2]))
                if len(plates) == 0:
                    print("No plates of "+move.capitalize()+" remain! Retry!")
                else:
                    plate = move.capitalize()+" ["+str(int(max(plates)))+"]"
                    print_hint2(player + " took PLATE "+plate+"!")
                    self.players[player].inventory.append(plate)
                    del self.game_inventory[self.game_inventory.index(plate)]
                    break
            elif move == "r":
                self.die_r()
                self.players[player].inventory.append("Diced")
                break
            elif move in ["o","d","w"]:
                self.OasisDesert(player,move)
                break
            elif move == "f":
                print("Which Camel do you want to set?")
                move = input()
                if move.capitalize() not in self.players[player].cards:
                    print(player + " doesn't have camel " + move.capitalize() + " (anymore)!")
                else:
                    print("Winner or Loser?\t\ttype w for winner and l for loser.")
                    move2 = input()
                    if move2 not in ["l","w"]:
                        print("Illegal type of bet! Retry!")
                    elif move2 == "l":
                        print_hint2(player + " has bet "+move.capitalize()+" as the game loser!")
                        del self.players[player].cards[self.players[player].cards.index(move.capitalize())]
                        self.game_loser.append([player,move.capitalize()])
                        break
                    elif move2 == "w":
                        print_hint2(player + " has bet "+move.capitalize()+" as the game WINNER!")
                        del self.players[player].cards[self.players[player].cards.index(move.capitalize())]
                        self.game_winner.append([player,move.capitalize()])
                        break
            else:
                print(move +" is not a legal move. Check Choices!")
        if move in ["o","d","w","r","t"]:
            self.rec=True
        # self.rec=True

class Field():
    Camels = CamelUp.extended_Camels.copy()
    def __init__(self,field=[],players=[],moved=[]):
        self.game_field = field
        self.players = players
        for i in range(len(self.game_field)):
            if len(self.game_field[i]) > 0:
                if self.game_field[i][0] not in self.Camels:
                    self.players[self.game_field[i][1]].plate_pos = i
        self.moved = moved
    def __repr__(self):
        pass

def render_field(field, players:dict):
    player_mask = {name: i+10 for i, name in enumerate(players.keys())}
    mask = CamelUp.mask
    mask["DESERT"] = 8
    mask["OASIS"] = 9

    rendered_field = np.zeros((len(field),7),dtype=int)
    for i in range(len(field)):
        if len(field[i]) > 0:
            if field[i][0] in ["DESERT","OASIS"]:
                rendered_field[i,0] = mask[field[i][0]]
                rendered_field[i,1] = player_mask[field[i][1]]
            else:
                for j in range(len(field[i])):
                    rendered_field[i,j] = mask[field[i][j]]
    # Invert the player_mask dictionary to map back from plate payoffs to player names later
    inv_player_mask = {v: k for k, v in player_mask.items()}
    return rendered_field, inv_player_mask

## helper function to generate all dice permutations
@nb.njit(cache=True)
def _all_dice_permutations(draw_n_camels):
    n_dice_rolls = 3**draw_n_camels
    dice_rolls = np.zeros((n_dice_rolls,draw_n_camels),dtype=np.int64)
    for i in range(n_dice_rolls):
        for j in range(draw_n_camels):
            dice_rolls[i,j] = (i // (3**j)) % 3 + 1
    return dice_rolls

@nb.njit(cache=True)
def _all_camel_permutations(camels_not_thrown: np.ndarray):
    """
    Generates all possible orders (permutations) to draw camels_not_thrown without replacement.
    Uses a standard algorithm since Numba does not support itertools.permutations.
    """
    n_camels_not_thrown = len(camels_not_thrown)
    out_size = 1
    for i in range(2, n_camels_not_thrown + 1):
        out_size *= i
    result = np.empty((out_size, n_camels_not_thrown), dtype=np.int64)
    a = camels_not_thrown; n=n_camels_not_thrown; 
    c = np.zeros(n, dtype=np.int64)

    result[0] = a
    idx = 1

    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                tmp = a[0]
                a[0] = a[i]
                a[i] = tmp
            else:
                tmp = a[c[i]]
                a[c[i]] = a[i]
                a[i] = tmp

            result[idx] = a
            idx += 1

            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return result

def format_tables(probabilities: pd.DataFrame, payoffs: pd.DataFrame, game_inventory_list: list = [],extended: bool = False) -> str:
    # layout parameters
    row_label_width = 14
    col_width = 10   # includes the extra leading space
    camels = probabilities.index.tolist()

    lines_probabilities = []

    # ---------- Probabilities ----------
    header = (
        f"{'Probabilities':<{row_label_width}}"
        + "".join(f"{c:>{col_width-1}} " for c in camels)
    )
    lines_probabilities.append(header)

    for row in probabilities.columns:
        line = f"{row:<{row_label_width}}"
        for c in camels:
            val = probabilities.loc[c, row] * 100.0
            line += f"{val:>{col_width - 1}.2f}%"
        lines_probabilities.append(line)

    # ---------- Payoffs ----------
    lines_payoffs = []
    header = (
        f"{'Payoffs':<{row_label_width}}"
        + "".join(f"{c:>{col_width-1}} " for c in camels)
    )
    lines_payoffs.append(header)
    line_idx = 0
    for row in payoffs.columns:
        line = f"{row:<{row_label_width}}"
        for c in camels:
            val = payoffs.loc[c, row]
            # signed, aligned float
            plate_name = f"{c} [{row[0]}]"
            plate_there = plate_name in game_inventory_list
            if plate_there and row[0] == "2" and extended:
                if game_inventory_list.count(plate_name) == 1:
                    plate_there = False
            if not plate_there:
                line += f"\033[31m{val:>{col_width-1}.2f}\033[0m "
            else:
                line += f"{val:>{col_width-1}.2f} "
        lines_payoffs.append(line)
        line_idx += 1
    if extended:
        line = f"{row:<{row_label_width}}"
        for c in camels:
            val = payoffs.loc[c, row]
            # signed, aligned float
            plate_name = f"{c} [{row[0]}]"
            if plate_name not in game_inventory_list:
                line += f"\033[31m{val:>{col_width-1}.2f}\033[0m "
            else:
                line += f"{val:>{col_width-1}.2f} "
        lines_payoffs.append(line)

    return lines_probabilities, lines_payoffs

#
# Splitting sim_all_moves into digestible sub-routines using numba where possible. 
# Focus: Reduce compile time by reducing function complexity and size.
#

def _camel_index_maps(rendered_field:np.ndarray, camels_not_thrown:list):
    """
    Return initial row/col indices of all relevant camels.
    Not jitted; helper.
    """
    camel_row_idx_base = np.zeros(8, dtype=np.int64)
    camel_col_idx_base = np.zeros(8, dtype=np.int64)
    if 6 in camels_not_thrown:  # Extended version (7 camels)
        camel_ids = np.arange(1, 8)
    else:
        camel_ids = np.arange(1, 6)
    #print("rendered_field:",rendered_field)
    #print("camel_ids:",camel_ids)
    #print("camels_not_thrown:",camels_not_thrown)
    for i in camel_ids:
        pos = np.argwhere(rendered_field == i)
        try:
            camel_row_idx_base[i] = pos[0][0]
            camel_col_idx_base[i] = pos[0][1]
        except Exception as e:
            print(rendered_field)
            print(i)
            print(pos)
            print(camel_ids)
            print(e)
            raise Exception("Unexpected error in _camel_index_maps")
    return camel_row_idx_base, camel_col_idx_base

@nb.njit(cache=True, parallel=True)
def _simulate_paths(
    rendered_field, all_camel_permutations, all_dice_permutations,
    camel_row_idx_base, camel_col_idx_base,
    n_threads, n_players, extra_dim, positions_local, DO_hits_local
):
    """
    Simulate all paths for a given game state.
    """
    for path_idx in nb.prange(all_camel_permutations.shape[0] * all_dice_permutations.shape[0]):
        camel_order_idx = path_idx // all_dice_permutations.shape[0]
        dice_idx = path_idx % all_dice_permutations.shape[0]
        dice_rolls = all_dice_permutations[dice_idx]
        camel_order = all_camel_permutations[camel_order_idx]
        tid = nb.get_thread_id()

        field_copy = rendered_field.copy()
        camel_row_idx = camel_row_idx_base.copy()
        camel_col_idx = camel_col_idx_base.copy()

        for move_idx in range(len(dice_rolls)):
            move = dice_rolls[move_idx]
            camel = camel_order[move_idx]
            row = camel_row_idx[camel]
            col = camel_col_idx[camel]
            end = (field_copy[row] != 0).sum()
            stack = field_copy[row, col:end].copy().reshape(1, -1)
            field_copy[row, col:end] = 0
            below_target = False

            if camel not in [6, 7]:
                target_field = row + move
                if field_copy[target_field, 0] == 8:
                    DO_hits_local[tid, field_copy[target_field, 1] - 10, 0] += 1
                    target_field -= 1
                    below_target = True
                elif field_copy[target_field, 0] == 9:
                    DO_hits_local[tid, field_copy[target_field, 1] - 10, 0] += 1
                    target_field += 1
            else:
                target_field = row - move
                if field_copy[target_field, 0] == 8:
                    DO_hits_local[tid, field_copy[target_field, 1] - 10, 0] += 1
                    target_field += 1
                    below_target = True
                elif field_copy[target_field, 0] == 9:
                    DO_hits_local[tid, field_copy[target_field, 1] - 10, 0] += 1
                    target_field -= 1
            if target_field < 0:
                target_field = 0
            stack_len = stack.shape[1]
            if below_target:
                end = (field_copy[target_field] != 0).sum()
                for j in range(end + stack_len - 1, stack_len - 1, -1):
                    field_copy[target_field, j] = field_copy[target_field, j - stack_len]
                for j in range(stack_len):
                    field_copy[target_field, j] = stack[0, j]
            else:
                end_target_field = (field_copy[target_field] != 0).sum()
                if end_target_field + stack_len > 7:
                    print("Warning: Stack length exceeds field length at target field:",target_field)
                    print("stack:",stack, "Camel moving", camel)
                    print(field_copy)

                field_copy[target_field, end_target_field: end_target_field + stack_len] = stack

            for i in range(7):
                if field_copy[target_field, i] == 0:
                    break
                camel_update = field_copy[target_field, i]
                camel_row_idx[camel_update] = target_field
                camel_col_idx[camel_update] = i

            if target_field > 15:
                break
        # Ranking and counting step
        camel_positions = np.zeros(5)
        for camel_rank in range(1, 6):
            row = camel_row_idx[camel_rank]
            col = camel_col_idx[camel_rank]
            camel_positions[camel_rank - 1] = row * 7 + col
        camel_ranking = np.argsort(-camel_positions.flatten())
        extra_d = 3 * camel_order[0] + dice_rolls[0] - 4
        positions_local[extra_d, tid, camel_ranking[0], 0] += 1
        positions_local[extra_d, tid, camel_ranking[1], 1] += 1
        for i in range(2, 5):
            positions_local[extra_d, tid, camel_ranking[i], 2] += 1

@nb.njit(cache=True, parallel=True)
def _aggregate_results(
    positions_local, DO_hits_local, game_inventory_matrix, n_players
):
    positions = positions_local.sum(axis=1)
    DO_hits = DO_hits_local.sum(axis=0)
    game_inventory_multiplier = np.empty((positions.shape[0], game_inventory_matrix.shape[0], game_inventory_matrix.shape[1]))
    for i in range(positions.shape[0]):
        game_inventory_multiplier[i] = game_inventory_matrix

    n_paths = positions.sum(axis=2)[:, 0]
    VOI_array = compute_voi_array(positions)
    VOI_array = VOI_array * game_inventory_multiplier
    tmp = VOI_array.sum(axis=2)
    next_voi = tmp.sum(axis=1)
    return positions, DO_hits, n_paths, next_voi, game_inventory_multiplier

@nb.njit(cache=True, parallel=True)
def _compute_now_voi(positions, game_inventory_matrix, n_paths):
    VOI_array_now = np.zeros((5,3), dtype=np.float64)
    B = np.array([[5,3,2],[1,1,1],[-1,-1,-1]])
    agg_positions = positions.sum(axis=0)
    inv_n = 1.0 / n_paths.sum()
    for i in nb.prange(5):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += agg_positions[i, k] * inv_n * B[k, j]
            if s < 1:
                s = 0
            VOI_array_now[i, j] = s
    VOI_array_now = VOI_array_now * game_inventory_matrix
    now_voi = VOI_array_now.sum()
    agg_positions = None  # Help out numba/gc
    return now_voi

def sim_all_moves(
    rendered_field: np.ndarray,
    n_players: int,
    n_camels_thrown: int,
    camels_not_thrown: list,
    game_inventory_matrix: np.ndarray,
    n_threads: int = 8,
    verbose: bool = True,
):
    '''
    Split implementation for reduced jit compile time.
    '''
    len_all_camels = n_camels_thrown + len(camels_not_thrown)
    draw_n_camels = len(camels_not_thrown)
    if len_all_camels > 5:
        draw_n_camels -= 1
    if draw_n_camels <= 0:
        if verbose:
            print("No more camels can be drawn (draw_n_camels == 0).")
        return np.zeros((5, 3), dtype=np.float64), np.zeros((n_players, 1), dtype=np.float64), 0.0

    all_dice_permutations = _all_dice_permutations(draw_n_camels)
    camel_permutations = _all_camel_permutations(np.array(camels_not_thrown, dtype=np.int64))
    if verbose:
        print("draw_n_camels: ", draw_n_camels)
        print("Number of paths: ", len(all_dice_permutations), "first 5 paths: ", all_dice_permutations[:5], "SHOULD number of paths", 3 ** draw_n_camels)
        print("Number of permutations: ", len(camel_permutations), "first 5 permutations:", camel_permutations[:5], "should permutations")
    camels_swap_buf = list(camels_not_thrown)
    if 6 in camels_not_thrown:
        camels_swap_buf.remove(6)
        camels_swap_buf.append(7)
        cmp2 = _all_camel_permutations(np.array(camels_swap_buf, dtype=np.int64))
        all_camel_permutations = np.concatenate((camel_permutations, cmp2), axis=0)
        camels_swap_buf.remove(7)
        camels_swap_buf.append(6)
    else:
        all_camel_permutations = camel_permutations
    if verbose:
        print("Number of full permutations: ", all_camel_permutations.shape[0] * all_dice_permutations.shape[0])
    camel_row_idx_base, camel_col_idx_base = _camel_index_maps(rendered_field, camels_not_thrown)
    n_perm = len(all_camel_permutations)
    if len_all_camels == 6:
        extra_dim = 3 * 7
    else:
        extra_dim = 3 * 5
    positions_local = np.zeros((extra_dim, n_threads, 5, 3), dtype=np.float64)
    DO_hits_local = np.zeros((n_threads, n_players, 1), dtype=np.float64)
    _simulate_paths(
        rendered_field, 
        all_camel_permutations, 
        all_dice_permutations, 
        camel_row_idx_base, camel_col_idx_base,
        n_threads, n_players, extra_dim, 
        positions_local, DO_hits_local
    )
    fame_inventory_matrix = game_inventory_matrix.copy()
    positions, DO_hits, n_paths, next_voi, game_inventory_multiplier = _aggregate_results(
        positions_local, DO_hits_local, game_inventory_matrix, n_players)
    inv_n = 1.0 / n_paths.sum()
    now_voi = _compute_now_voi(positions, game_inventory_matrix, n_paths)
    VOI = ((next_voi - now_voi) * n_paths).sum() / n_paths.sum()
    return (positions * inv_n).sum(axis=0), DO_hits * inv_n, VOI

@nb.njit(cache=True, parallel=True)
#@nb.njit(parallel =True)
def compute_voi_array(positions):
    n = positions.shape[0]
    VOI = np.zeros((n, 5, 3), dtype=np.float64)

    score = np.array([[5.0, 3.0, 2.0],
                      [1.0, 1.0, 1.0],
                      [-1.0, -1.0, -1.0]])

    for i in nb.prange(n):
        n_paths = positions[i, 0, :].sum()
        if n_paths == 0:
            continue

        inv_n = 1.0 / n_paths

        for r in range(5):
            for c in range(3):
                s = 0.0
                for k in range(3):
                    s += positions[i, r, k] * inv_n * score[k, c]
                if s < 1:
                    s = 0
                VOI[i, r, c] = s

    return VOI

class player(): #for simulation
    def __init__(self,name):
        while name in CamelUp.standard_Camels:
            print("ILLEGAL PLAYER NAME! Can't be Camel Name! Retry:")
            name = input()
        self.cards = copy.deepcopy(CamelUp.standard_Camels)
        self.name = name
        self.coins = 3
        self.inventory = []
        self.inventory_payoffs = []
        self.expected_payoff = 0
        self.plate_pos = None
        self.plate_value = 0
    def end_of_round(self):
        self.coins += self.expected_payoff
        print_hint2(self.name + " earned " + str(int(self.expected_payoff)) +" this turn.")
        self.inventory = []
        self.inventory_payoffs = []
        self.expected_payoff = 0
        self.plate_pos = None
        self.plate_value = 0

"""
Field Design
——————————————————————————————————————————————————————————————————
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
——————————————————————————————————————————————————————————————————
|  xyz--zyx  |        <16>     < 1>     < 2>        |  xyz--zyx  |
|  xyz--zyx  |   <15>                        < 3>   |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  | <14>                            < 4> |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
——————————————                                      ——————————————
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  | <13>                            < 5> |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
——————————————                                      ——————————————
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  | <12>                            < 6> |  xyz--zyx  |
|  xyz--zyx  |                                      |  xyz--zyx  |
|  xyz--zyx  |   <11>                        < 7>   |  xyz--zyx  |
|  xyz--zyx  |        <10>     < 9>     < 8>        |  xyz--zyx  |
——————————————————————————————————————————————————————————————————
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
|  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |  xyz--zyx  |
——————————————————————————————————————————————————————————————————

Center Design

        <16>     < 1>     < 2>        |
   <15>                        < 3>   |
                                      |
 <14>                            < 4> |
                                      |
                                      —
                                      |
                                      |
 <13>                            < 5> |
                                      |
                                      |
                                      —
                                      |
 <12>                            < 6> |
                                      |
   <11>                        < 7>   |
        <10>     < 9>     < 8>        |
———————————————————————————————————————
"""

f'''
New Field Design (7 Camels)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
║      xyz----zyx      ║              <16>                < 1>              < 2>            ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║      <15>                                                < 3>      ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║   <14>                                                       < 4>  ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
═════════════════════════                                                                  ═════════════════════════
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║   <13>                                                       < 5>  ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
═════════════════════════                                                                  ═════════════════════════
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║   <12>                                                       < 6>  ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║      <11>                                                < 7>      ║      xyz----zyx      ║
║      xyz----zyx      ║                                                                    ║      xyz----zyx      ║
║      xyz----zyx      ║              <10>                < 9>              < 8>            ║      xyz----zyx      ║
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║      xyz----zyx      ║
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

'''
print()





