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

class CamelUp():
    '''
    Main Class for Camel Cup calculations.
    This class can run a game and simulate the paths in each round.
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
    standard_Camels = ["Yellow","Blue","Green","Orange","Purple"]
    standard_Inventory = [] # contains 
    for i in standard_Camels:
        standard_Inventory.extend([i+" [5]",i+" [3]",i+" [2]"])
    field_structure = [[14,15,0,1,2],
                       [13,None,None,None,3],
                       [12,None,None,None,4],
                       [11,None,None,None,5],
                       [10,9,8,7,6]]
    center_design = ['        <16>     < 1>     < 2>        |',
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
    print_dim = [31,120]
    mask = {"Purple":1,"Blue":2,"Orange":3,"Yellow":4,"Green":5, "White":6, "Black":7}
    render_field_cell_width = 12
    parallel_workers= 8
    def __init__(self,
                 n_players:int,
                 start:bool = True,
                 field = "",
                 tutorial=True,
                 black_white= False):
        self.black_white = black_white
        self.Camels = copy.deepcopy(self.standard_Camels)
        self.Inventory = copy.deepcopy(self.standard_Inventory)
        if self.black_white:
            self.Camels.append("Black")
            self.Camels.append("White")
            self.Inventory.extend([f"{i} [2]" for i in self.standard_Camels])
        self.total_width = (5*self.render_field_cell_width+6)
        self.rec = True
        self.tutorial=tutorial
        self.fields = {}
        self.game_winner = []
        self.game_loser = []
        self.moved = []
        self.VOI = 0
        self.win_prob = {}
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
        print("—"*76)
        self.print_game(True, False)
        self.print_c()
        print("—"*76)
    def position(self,start=False):
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
    def OasisDesert(self,player,plate = ""): 
        if player not in self.players.keys():
            print("\n!!!!!!!!!!!!!!!!!!!!\nInvalid player name!\n!!!!!!!!!!!!!!!!!!!!\n")
            return 0
        for i in range(len(self.game_field)):
            if player in self.game_field[i]:
                self.game_field[i] = []
                self.players[player].plate_pos = None
                break
        if plate == "":
            print(player+"'s Oasis/Desert on the field?\n 'o' for OASIS, 'd' for DESERT, "+\
                  "anything else to WITHDRAW plate")
            plate = input()
        else:
            plate = plate.capitalize()
        if plate == "O":
            plate = "OASIS"
        elif plate == "D":
            plate = "DESERT"
        else:
            plate = "WITHDRAW"
        if plate in ["OASIS","DESERT","WITHDRAW"]:
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
                   ):
        self.rendered_output = [""]*self.print_dim[0]
        self.rendered_header = [""]*3
        if field:
            self.render_field()
        if payoffs:
            if self.win_prob == {}:
                self.one_turn(print_option=False,OD=False,player = list(self.players.keys())[0])
            self.render_payoffs()
        print("\n"+"\n".join(self.rendered_header)+"\n")
        print("\n".join(self.rendered_output))
        
    def render_field(self):
        total_width = self.total_width
        gap_margin = self.gap_margin
        '''
        Renders field data into formatted output.
        Uses self.game_field and self.field_structure 
         and the natural shape of the CamelUp to render 
         the field output into a 31x66 pixel image
        '''
        
        for row in range(len(self.rendered_output)):
            self.rendered_output[row]+=" "*gap_margin
        for row in range(len(self.rendered_header)):
            self.rendered_header[row]+=" "*gap_margin
        header_statement = "Current Field"
        self.rendered_header[0] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = total_width)
        self.rendered_header[1] += "{string:^{width}s}".format(\
                                    string="##  "+header_statement+"  ##",
                                    width = total_width)
        self.rendered_header[2] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = total_width)
        self.rendered_output[0]+= "—"*total_width
        local_center_design = self.center_design+[]
        local_center_design[8] = local_center_design[8].format(VOI=self.VOI)
        for row_n in range(5):
            for column_n in range(5):
                if row_n in [1,2,3] and column_n in [1,2,3]:
                    if column_n ==1:
                        for render_row in range(row_n*6,row_n*6+6):
                            self.rendered_output[render_row+1]+=local_center_design[render_row-6]
                    continue
                field_n = self.field_structure[row_n][column_n]
                field_n_content = copy.deepcopy(self.game_field[field_n])
                field_contents = [" "*self.render_field_cell_width+"|"]*5+\
                    ["—"*(self.render_field_cell_width+1)]
                if str(field_n+1)+"O" in self.fields.keys() or\
                    str(field_n+1)+"D" in self.fields.keys():
                        if field_n_content == []:
                            field_n_content = ["(DESERT)",self.fields[str(field_n+1)+"D"],
                                               "(OASIS)",self.fields[str(field_n+1)+"O"]]
                        elif field_n_content[0] in ["DESERT","OASIS"]:
                            field_n_content +=[np.nan,"",""]
                            if field_n_content[0] == "OASIS":
                                field_n_content[2]  = self.fields[str(field_n+1)+"O"]
                                field_n_content[3]  = "({string:.2f})".\
                                    format(string=self.fields[str(field_n+1)+"D"])
                            else:
                                field_n_content[2]  = self.fields[str(field_n+1)+"D"]
                                field_n_content[3]  = "({string:.2f})".\
                                    format(string=self.fields[str(field_n+1)+"O"])
                            if "W"+str(field_n+1) in self.fields.keys():
                                field_n_content[4]  = "(({string:.2f}))".\
                                    format(string=self.fields["W"+str(field_n+1)])
                if field_n_content == []:
                    pass
                elif field_n_content[0] in ["DESERT","OASIS"]:
                    field_contents[0] = "{string:^{width}s}|".\
                        format(string=field_n_content[0],
                               width=self.render_field_cell_width)
                    field_contents[1] = "{string:^{width}s}|".\
                        format(string=field_n_content[1],
                               width=self.render_field_cell_width)
                    if len(field_n_content)>2:
                        field_contents[2] = "{string:^{fill}{width}f}|".\
                            format(string=field_n_content[2],fill=" ",
                                   width=str(self.render_field_cell_width)+".2")
                    if len(field_n_content)>3:
                        field_contents[3] = "{string:^{width}s}|".\
                            format(string=field_n_content[3],
                                   width=str(self.render_field_cell_width))
                    if len(field_n_content)>3:
                        field_contents[4] = "{string:^{width}s}|".\
                            format(string=field_n_content[4],
                                   width=str(self.render_field_cell_width))        
                elif field_n_content[0] == "(DESERT)":
                    field_contents[1] = "{string:^{width}s}|".\
                        format(string=field_n_content[0],
                               width=self.render_field_cell_width)
                    field_contents[2] = "{string:^{fill}{width}f}|".\
                            format(string=field_n_content[1],fill=" ",
                                   width=str(self.render_field_cell_width)+".2")
                    field_contents[3] = "{string:^{width}s}|".\
                        format(string=field_n_content[2],
                               width=str(self.render_field_cell_width))
                    field_contents[4] = "{string:^{fill}{width}f}|".\
                            format(string=field_n_content[3],fill=" ",
                                   width=str(self.render_field_cell_width)+".2")
                else:
                    start = 2-len(field_n_content)//2
                    for row_o in range(start,start+len(field_n_content)):
                        # print(field_contents,
                        #       start,
                        #       row_o,
                        #       field_n_content[start-row_o-1],
                        #       self.render_field_cell_width)
                        if field_n_content[start-row_o-1] in self.moved:
                            field_contents[row_o] = "{string:^{width}s}|".\
                                    format(string="["+field_n_content[start-row_o-1]+"]",
                                           width=self.render_field_cell_width)
                        else:
                            field_contents[row_o] = "{string:^{width}s}|".\
                                    format(string=field_n_content[start-row_o-1],
                                           width=self.render_field_cell_width)
                for row_m in range(6):
                    render_row = row_m+row_n*6+1
                    if column_n == 0:
                        extra_sign = "|"
                        if row_m == 5:
                            extra_sign = "—"
                        self.rendered_output[render_row]+= extra_sign
                    self.rendered_output[render_row]+= field_contents[row_m]
            
    def render_payoffs(self):    
        gap_margin = self.gap_margin
        cell_width = 10
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
        
        for row in range(len(self.rendered_output)):
            self.rendered_output[row]+=" "*gap_margin
        self.rendered_output[0]+= "—"*(cell_width*4+5)
        self.rendered_output[1]+= "|"+"|".join(["{i:^{width}}".format(i=i,width = cell_width) \
                                                for i in [" ","First","Second","Lose"]])+"|"
        self.rendered_output[2]+= "—"*(cell_width*4+5)
        for camel_n in range(len(self.Camels)):
            camel = self.Camels[camel_n]
            row = "|{camel:^{width}s}|".format(camel=camel,width=cell_width)+\
                "|".join(["{i:^{width}.1%}".format(i=i,width = cell_width) \
                                                for i in [self.win_prob[camel],
                                                          self.sec_prob[camel],
                                                          self.lose_prob[camel]]])+"|"
            self.rendered_output[2+camel_n*2+1] += row
            self.rendered_output[2+camel_n*2+2]+= "—"*(cell_width*4+5)
        row = 2+camel_n*2+3
        self.rendered_output[row] += " "*(cell_width*4+5)
        header_statement = "Expected Payoffs"
        self.rendered_output[row+1] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = width-2*gap_margin)
        self.rendered_output[row+2] += "{string:^{width}s}".format(\
                                    string="##  "+header_statement+"  ##",
                                    width = width-2*gap_margin)
        self.rendered_output[row+3] += "{string:^{width}s}".format(\
                                    string="#"*(len(header_statement)+2*4),
                                    width = width-2*gap_margin)
        self.rendered_output[row+4] += " "*(cell_width*4+5)
        row = row+5
        self.rendered_output[row]+= "—"*(cell_width*4+5)
        self.rendered_output[row+1]+= "|"+"|".join(["{i:^{width}}".format(i=i,width = cell_width) \
                                                for i in [" ","5-Plate","3-Plate","2-Plate"]])+"|"
        self.rendered_output[row+2]+= "—"*(cell_width*4+5)
        for camel_n in range(len(self.Camels)):
            camel = self.Camels[camel_n]
            row_text = "|{camel:^{width}s}|".format(camel=camel,width=cell_width)
            for win_value in [5,3,2]:
                brackets = ["",""]
                if camel + " [{:d}]".format(win_value) not in self.game_inventory:
                    brackets = ["[","]"]
                row_text += "{value:^{width}s}".format(value = brackets[0]+\
                    str(round(self.ret[camel+" [{:d}]".format(win_value)],2))+"$"+brackets[1],
                    width = cell_width)+"|"
            self.rendered_output[camel_n*2+3+row] += row_text
            self.rendered_output[camel_n*2+4+row]+= "—"*(cell_width*4+5)
        
    def print_c(self):
        print_hint2("Coins of players:")
        max_len_name = 10
        for i in self.players.values():
            max_len_name = max(max_len_name,len(i.name))
        print_coins = "Player:          | "
        for i in self.players.values():
            print_coins += print_adj(i.name,max_len_name,"c") +" | "
        print_coins = print_coins[:-3] +"\n"+"–"*(19+len(self.players)*(max_len_name+3))+"\nCoins:           | "
        for i in self.players.values():
            print_coins += print_adj(str(i.coins),max_len_name,"c") +" | "
        print_coins = print_coins[:-3] +"\nExpected Payoff: | "
        for i in self.players.values():
            print_coins += print_adj(str(i.expected_payoff),max_len_name,"c") +" | "
        print_coins = print_coins[:-3] +"\n"
        print(print_coins)
    def moved_f(self,camel):
        if camel not in self.moved and camel in self.Camels:
            self.moved.append(camel)
    def cl(self):
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
        self.win_prob = {}
        self.fields = {}
    def move(self,camel,steps):
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
                field = j + steps
                break
        if "OASIS" in self.game_field[field]:
            player = self.game_field[field][1]
            print("#"*len(player)+"#############\n"+player+" gets a coin!\n"+"#"*len(player)+"#############\n")
            self.players[self.game_field[field][1]].coins+=1
            field += 1
            self.game_field[field].extend(moving_camels)
        elif "DESERT" in self.game_field[field]:
            player = self.game_field[field][1]
            print("#"*len(player)+"#############\n"+player+" gets a coin!\n"+"#"*len(player)+"#############\n")
            self.players[self.game_field[field][1]].coins+=1
            field -= 1
            moving_camels.extend(self.game_field[field])
            self.game_field[field] = moving_camels
        else:
            self.game_field[field].extend(moving_camels)
    def value_of_information_increase(self,VOI):
        '''
        This function is a bit buggy, interpretation needs to be added for this.
        
        '''
        AVG_inf     = 0
        iterations  = 0
        for i in self.Camels:
            if i not in self.moved:
                for j in range(1,4):
                    payoff = {}
                    for k in range(len(self.game_field)):
                        if "OASIS" in self.game_field[k] or "DESERT" in self.game_field[k]:
                            payoff[str(k+1)] = 0
                    npaths = 0
                    first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
                    second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
                    # print(i+" "+str(j))
                    field,hit = self.move_simulation(copy.deepcopy(self.game_field), i,j)
                    if len([*field[16],*field[17],*field[18]]) > 0:
                        ranks = self.rank(field)
                        first[ranks[-1]]+=1*3**(4-len(self.moved))
                        second[ranks[-2]]+=1*3**(4-len(self.moved))
                        if len(hit.values())>0:
                            key = list(hit.keys())[0]
                            payoff[key]+=hit[key]
                        npaths += 1*3**(4-len(self.moved))
                        continue
                    liste = []
                    for k in self.Camels:
                        if k != i and k not in self.moved:
                            liste.append(k)
                    first,second,payoff,n_paths1 = self.flexible_for(liste,field,first,second,payoff)
                    if len(hit.values())>0:
                        key = list(hit.keys())[0]
                        payoff[key]+=hit[key]*n_paths1
                    npaths += n_paths1
                    for k in self.Camels:
                       for m in [5,3,2]:
                           if k+" ["+str(m)+"]" in self.game_inventory:
                               e_payoff = first[k]/npaths*m+second[k]/npaths-(1-(first[k]-second[k])/npaths)
                               if e_payoff>=1:
                                   AVG_inf+=e_payoff
                    iterations+=1
        return AVG_inf/iterations-VOI
    def one_turn(self,print_option=True,OD=False,player = ""):
        '''
        This function manages all the payoffs
        '''
        first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
        second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
        # determine camels that haven't moved:
        Camels_die = copy.deepcopy(self.Camels)
        for i in self.moved:
            if i in Camels_die:
                del Camels_die[Camels_die.index(i)]

        ## encode camels that still die
        Camels_die_rendered = [self.mask[i] for i in Camels_die]
        ## handle special case of black and white camel
        if "Black" in self.Camels:
            if 6 in Camels_die_rendered and 7 in Camels_die_rendered:
                Camels_die_rendered.remove(7)
            elif 6 not in Camels_die_rendered and 7 in Camels_die_rendered:
                Camels_die_rendered.remove(7)
            else:
                Camels_die_rendered.remove(6)
            n_camels_thrown = 6-len(Camels_die_rendered)
        else:
            n_camels_thrown = 5-len(Camels_die_rendered)

        # determine current field of play
        start_field = copy.deepcopy(self.game_field)
        ## render said field
        rendered_field, player_mapping = render_field(self.game_field)
        ## run simulation numba accelerated
        base_positions, base_DO_hits = sim_all_moves(rendered_field, len(self.players), n_camels_thrown, Camels_die_rendered, verbose=False)
        
        payoff = {}
        n_paths = base_positions[0,:].sum()
        
        for i in range(rendered_field.shape[0]):
            if rendered_field[i, 0]>7:
                player_index = rendered_field[i, 1]
                player_name = player_mapping[player_index] ## player position maps from player index + 10 to player name
                payoff[player_name] = {
                    "Type": "Desert" if rendered_field[i, 0] == 8 else "Oasis", 
                    "Field": i,
                    "Expected Payoff": base_DO_hits[player_index-10, 0]/n_paths}

        base_probabilities = base_positions/n_paths
        base_payoffs = np.matmul(base_probabilities, np.array([[5,3,2],[1,1,1],[-1,-1,-1]]))
        base_payoffs = pd.DataFrame(base_payoffs, index=self.Camels[:5], columns=["5-Plate", "3-Plate", "2-Plate"])

        ## handle desert value calculation for voi and differentials for desert plate changes
        ## !!! markter
        
        desert_value = {}
        for i in payoff.keys():
            desert_value[i] = payoff[i]/n_paths
        VOI = 0
        for i in self.game_inventory:
            if self.ret[i]>=1:
                VOI+=self.ret[i]
        if len(self.moved)<4:
            self.VOI = self.value_of_information_increase(VOI)
        else:
            self.VOI = 1
        if print_option:
            self.print_game(False,True)
        for i in self.players.keys():
            e_payoff = 0
            for j in self.players[i].inventory:
                if j == "Diced":
                    e_payoff+=1
                else:
                    e_payoff += self.ret[j]
            if self.players[i].plate_pos != None:
                e_payoff += desert_value[str(self.players[i].plate_pos+1)]
            self.players[i].expected_payoff = round(e_payoff,2)
        if OD:
            # classify camels on who has their plates
            players_inventory = []
            for i in self.players.keys():
                for j in self.players[i].inventory:
                    if j != "Diced":
                        players_inventory.append([j,i])
            ## determine position of players desert/oasis fields
            pos = None
            if player in self.players.keys():
                pos = self.players[player].plate_pos
            ## determine value of withdrawls
            # if self.rec:
            fields = self._desert_iterator(player)
            # self.fields = fields|{}
            # else:
            #     fields = self.fields|{}
            ## iterate over results of desert iterator:
            for i in fields.keys():
                if "W" in i:
                    value = 0
                elif i == "xxx":
                    fields[i] = 0
                    continue
                else:
                    value = fields[i][0][i[:-1]] #coin value of field
                player_i = None
                if i[0]!="W":
                    game_field_i = int(i[:-1])-1
                else:
                    game_field_i = int(i[1:])-1
                if len(self.game_field[game_field_i]) >0:
                    if self.game_field[game_field_i][0] in ["DESERT","OASIS"]:
                        player_i =  self.game_field[game_field_i][1]
                for j in desert_value.keys():
                    if int(j) > game_field_i:
                        continue
                    if player_i == self.game_field[int(j)-1][1] or \
                        player_i == player and j!= i[0]:
                        pass
                    else:
                        # print( desert_value, j, i, fields) ##testing
                        if j in fields[i][0].keys():
                            value+= desert_value[j]-fields[i][0][j]
                        else:
                            print("--",fields[i],j,i,desert_value,sep="\n--")
                if player_i is None:
                    player_i = player
                for j in players_inventory:
                    if player_i == j[1]:
                        value += fields[i][1][j[0]]-self.ret[j[0]]
                    else:
                        value += self.ret[j[0]]-fields[i][1][j[0]]
                fields[i] = value
            self.fields = {**fields,**{}}
            print_fields = "Plates: {"
            for i in fields.keys():
                if "W" in i:
                    print_fields+= " Withdrawal: " + str(round(fields[i],2)) +", "
                elif i[-1] == "D":
                    print_fields += "Desert "+i[:-1]+": " + str(round(fields[i],2)) +", "
                elif i[-1] == "O":
                    print_fields += "Oasis "+i[:-1]+": " + str(round(fields[i],2)) +", "
            print_hint2(print_fields[:-2]+" }")
            self.rec = False
    def move_simulation(self,field,camel,steps):
        for j in range(len(field)):
            if camel in field[j]:
                index = field[j].index(camel)
                moving_camels = field[j][index:]
                field[j] = field[j][:index]
                step = j + steps
                break
        hit = {}
        # print(camel + " "+str(step+1))
        if "OASIS" in field[step]:
            # player = field[step][1]
            # print("#"*len(player)+"#############\n"+" gets a coin!\n"+"#"*len(player)+"#############\n")
            hit[str(step+1)] = 1
            step += 1
            field[step].extend(moving_camels)
        elif "DESERT" in field[step]:
            # player = field[step][1]
            # print("#"*len(player)+"#############\n"+" gets a coin!\n"+"#"*len(player)+"#############\n")
            hit[str(step+1)] = 1
            step -= 1
            moving_camels.extend(field[step])
            field[step] = moving_camels
        else:
            field[step].extend(moving_camels)
        # print(field)
        return field,hit
    def _desert_iterator(self,player): 
        '''
        !!! Testing may be required to improve the performance here and make sure nothing computes 
        more than once.
        
        This function simulates the game for the value of the desert and oasis fields.
        Problem:
            Withdrawel of the desert or oasis fields is only possible for the current player
            and with his own plate. 
            Calculation at once for all of these withdrawels might not be necessary.
            Maybe safe a per player dictionary of values of withdrawel.

        Parameters
        ----------
        player : TYPE
            Player as reference. 

        Returns
        -------
        fields: dict
            Potential fields to place desert or oasis on.

        '''
        # import pdb; pdb.set_trace()
        first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
        second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"Purple":0}
        ## look for valuable fields for deserts
        field = self.game_field + []
        W_field = "xxx"
        j = 0 # count the maximum field
        for i in range(len(field)):
            if player in field[i]:
                W_field = "W"+str(i+1)
                field[i] = []
                break

        Camels = 0
        camel_found = False
        Camel_distance = [0] * 16
        for i in range(16):
            if field[i]==[]:
                distance +=1
            if set(field[i]) & set(self.Camels[:5]) and not camel_found:
                camel_found = True
                distance = 0
                Camel_distance[i] = distance
            elif camel_found:
                distance += 1
                Camel_distance[i] = distance
        if len(self.Camels) > 5:
            camel_found = False
            for i in range(15,-1,-1):
                if field[i] == ["Black"] or field[i] == ["White"] and not camel_found:
                    camel_found = True
                    distance = 0
                    Camel_distance[i] = min(distance,Camel_distance[i])
                elif camel_found:
                    distance += 1
                    Camel_distance[i] = min(distance,Camel_distance[i])
        legal_fields = [idx for idx,distance in enumerate(Camel_distance) if distance > 0 and distance < 5]

        Camels_die = [camel for camel in self.Camels if camel not in self.moved]
        payoff = {} 
        for i in range(len(field)):
            if "OASIS" in field[i] or "DESERT" in field[i]:
                payoff[str(i+1)] = 0
        ## field with removed oasis and desert value and or with plain field as is.
        if self.rec: 
            self.players_removal = {}
            fields = {}
        else:
            fields = self.game_fields|{}
        OASISret = {} #new oasis fields evaluation dictionary
        if W_field!= "xxx":
            if player not in self.players_removal.keys():
                removedFirst, removedSecond, removedPayoff, removedNpaths = \
                    self.flexible_for(Camels_die,
                                      field+[],
                                      {**first,**{}},
                                      {**second,**{}},
                                      {**payoff,**{}})
                for i in removedPayoff.keys():
                    removedPayoff[i]=removedPayoff[i]/removedNpaths
                self.players_removal[player] = {
                    "first":removedFirst,"second":removedSecond,"payoff":removedPayoff,
                    "npaths":removedNpaths}
            else:
                removedFirst = self.players_removal[player]["first"]
                removedSecond= self.players_removal[player]["second"]
                removedPayoff= self.players_removal[player]["payoff"]
                removedNpaths= self.players_removal[player]["npaths"]
                
            for i in self.Camels: 
                for k in [5,3,2]:
                    OASISret[i+" ["+str(k)+"]"] = \
                        ((k+1)*removedFirst[i]+2*removedSecond[i]-removedNpaths)/removedNpaths
            fields[W_field] = [removedPayoff,OASISret]
        if self.rec:
            field = self.game_field + []
            for j in range(start,end):
                if len(field[j]) == 0 or "DESERT" in field[j] or "OASIS" in field[j]:
                    if not ("OASIS" in field[j-1] or "OASIS" in field[j+1] or\
                        "DESERT" in field[j-1] or "DESERT" in field[j+1]):
                        field[j] = ["OASIS",player]
                        payoff = {}
                        for i in range(len(field)):
                            if "OASIS" in field[i] or "DESERT" in field[i]:
                                payoff[str(i+1)] = 0
                        OASISfirst, OASISsecond, OASISpayoff, OASISnpaths = \
                            self.flexible_for(Camels_die,copy.deepcopy(field),\
                                              copy.deepcopy(first),copy.deepcopy(second),\
                                              copy.deepcopy(payoff))
                        field[j] = ["DESERT",player]
                        payoff = {}
                        for i in range(len(field)):
                            if "OASIS" in field[i] or "DESERT" in field[i]:
                                payoff[str(i+1)] = 0
                        DESERTfirst, DESERTsecond, DESERTpayoff, DESERTnpaths = \
                            self.flexible_for(Camels_die,copy.deepcopy(field),\
                                              copy.deepcopy(first),copy.deepcopy(second),\
                                              copy.deepcopy(payoff))
                        for i in OASISpayoff.keys():
                            OASISpayoff[i]=OASISpayoff[i]/OASISnpaths
                        OASISret = {}
                        for i in self.Camels: 
                            for k in [5,3,2]:
                                OASISret[i+" ["+str(k)+"]"] = \
                                    ((k+1)*OASISfirst[i]+2*OASISsecond[i]-OASISnpaths)/OASISnpaths
                        fields[str(j+1)+"O"] = [OASISpayoff,OASISret]
                        for i in DESERTpayoff.keys():
                            DESERTpayoff[i]=DESERTpayoff[i]/DESERTnpaths
                        DESERTret = {}
                        for i in self.Camels: 
                            for k in [5,3,2]:
                                DESERTret[i+" ["+str(k)+"]"] = \
                                    ((k+1)*DESERTfirst[i]+2*DESERTsecond[i]-DESERTnpaths)/DESERTnpaths
                        fields[str(j+1)+"D"] = [DESERTpayoff,DESERTret]
                        # special case for testing: put in expected payoff
                        if self.game_field[j] == []:
                            field[j] = []
        self.game_fields = fields|{}    
        return fields # dictionary of potential plate fields: returns and their payoff matrix
    def rank(self,field):
        ranks = []
        for i in field:
            if "OASIS" not in i and "DESERT" not in i and len(i) > 0:
                ranks.extend(i)
        return ranks
    def die_r(self):
        moving = []
        for i in self.Camels:
            if i not in self.moved:
                moving.append(i)
        camel = moving[randrange(len(moving))]
        steps = randrange(1,4)
        lenstr = 6+7+7+1+len(camel)
        print("\n"+"#"*lenstr+"\nCamel "+camel+" moves "+str(steps)+" steps!\n"+"#"*lenstr+"\n")
        self.move(camel,steps)
    def game_not_end(self):
        if len([*self.game_field[16],*self.game_field[17],*self.game_field[18]]) == 0:
            return True
        else:
            return False
    def game(self,player = ""):
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
            index += 1
        self.game_end() #end of game
    def game_end(self):
        self.moved = self.Camels
        self.cl()
        self.print_c()
        for i in self.game_field:
            if i != []:
                if i[0] not in ["OASIS","DESERT"]:
                    loser = i[0]
                    break
        winner = [*self.game_field[16],*self.game_field[17],*self.game_field[18]][-1]
        print_hint2(loser+" lost track and is dead last!")
        j = 0
        words = ["first","second","third","fourth","fifth","sixth"]
        coins = ["8","5","3","2","1","no"]
        for i in self.game_loser:
            if i[1] == loser:
                print(i[0]+" predicted the loser correctly "+words[j]+" and gets "+coins[j],end = " ")
                if j == 0:
                    print("coins. Congratulations!")
                elif j != 4:
                    print("coins.")
                elif j == 4:
                    print("coin.")
                if j < 5:
                    self.players[i[0]].coins+=int(coins[j])
                j+=1
            else:
                print(i[0]+" wrongly predicted the losing camel and loses a coin.")
                self.players[i[0]].coins-=1
        j=0
        print_hint2(winner+" won the race!!!!")
        for i in self.game_winner:
            if i[1] == winner:
                print(i[0]+" predicted the winner correctly "+words[j]+" and gets "+coins[j],end = " ")
                if j == 0:
                    print("coins. Congratulations!")
                elif j != 4:
                    print("coins.")
                elif j == 4:
                    print("coin.")
                if j < 5:
                    self.players[i[0]].coins+=int(coins[j])
                j+=1
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
            print_hint2("The winners are "+print_win)
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
              print_adj("Set Oasis/Desert:",col0_len)+" | "+print_adj("Take final bet:",col0_len)+"\n"+\
                  "–"*(10+5*col0_len))
        print(print_adj("Type",10,"c")+" | "+print_adj("color",col0_len-5,"c")+" | "+\
              print_adj("t",col0_len-5,"c")+" | "+ print_adj("r",col0_len,"c")+" | "+
              print_adj("o, d or w",col0_len,"c")+" | "+print_adj("f",col0_len,"c"))
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
        self.print_i()
        self.print_c()
        print_hint2("Camels not moved: "+CNM[:-2])
        print_hint2(player+"'s move")
        # if self.tutorial:
        #     print("\nType:\n\
        #            \t{throw:20s}for throwing the dice.\n\
        #            \t{camel_name:20s}for taking a round bet on a camel\n\
        #            \t{random:20s}for simulating a die throw (camel move)\n\
        #            \t{oasis:20s}for moving the Oasis or Desert plate or Withdrawing\n\
        #            \t{final:20s}for taking a game bet\n\
        #            \t{exit_s:20s}for exiting the game\n\
        #            ".format(throw = "t",camel_name="Name of Camel",
        #                     random="r",oasis="o, d, w", 
        #                     final="f",exit_s="exit"))
        while True:
            print("Move t for throw a camel, r for random throw, o or d or w for desert, oasis, [Camel] for bet, f for final bet")
            move = input()
            if move == "t":
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

def render_field(Field):
    player_mask = {name: i+10 for i, name in enumerate(Field.players.keys())}
    mask = CamelUp.mask
    mask["DESERT"] = 8
    mask["OASIS"] = 9

    rendered_field = np.zeros((len(Field.game_field),7),dtype=int)
    for i in range(len(Field.game_field)):
        if len(Field.game_field[i]) > 0:
            if Field.game_field[i][0] in ["DESERT","OASIS"]:
                rendered_field[i,0] = mask[Field.game_field[i][0]]
                rendered_field[i,1] = player_mask[Field.game_field[i][1]]
            else:
                for j in range(len(Field.game_field[i])):
                    rendered_field[i,j] = mask[Field.game_field[i][j]]
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



@nb.njit(cache=True, parallel=True)
def sim_all_moves(
    rendered_field:np.ndarray, n_players:int, n_camels_thrown:int, 
    camels_not_thrown:list[int], verbose:bool = True
       ):
    '''
    This function simulates the paths for the moves of the camels
    Parameters
    ----------
    rendered_field: np.ndarray
        The rendered field, returned by the render_field function.
    n_players: int
        Number of players in the game.
    n_camels_thrown: int
        Number of camels that have been thrown.
    camels_not_thrown: list[int]
        Camels that have not been thrown. Information needed to simulate properly and to infer game version.

    Returns
    -------
    probabilities: array of shape (5,3)
        Probabilities of the camels to end up in the first, second, third and lower positions.
        The first dimension is the camel, the second dimension is the position.
    payoffs: array of shape (x,2)
        Positions and hits for all desert and oasis fields.
    '''

    len_all_camels = n_camels_thrown + len(camels_not_thrown)
    draw_n_camels = len(camels_not_thrown)
    if len_all_camels > 5:
        draw_n_camels -= 1
    # Another early exit for <=0 camels to move
    if draw_n_camels <= 0:
        if verbose:
            print("No more camels can be drawn (draw_n_camels == 0).")
        return np.zeros((5, 3), dtype=np.int64), np.zeros((n_players, 3), dtype=np.int64)

    all_dice_permutations = _all_dice_permutations(draw_n_camels)
    # Generate all possible permutations of camels_not_thrown (numba friendly: i.e., as lists of integer arrays)
    # (This is separate from all_dice_permutations, which is all product/dice rolls)
    camel_permutations = _all_camel_permutations(np.array(camels_not_thrown, dtype=np.int64))

    if verbose:
        print("draw_n_camels: ", draw_n_camels)
        print("Number of paths: ", len(all_dice_permutations), "first 5 paths: ", all_dice_permutations[:5], "SHOULD number of paths", 3**draw_n_camels)
        print("Number of permutations: ", len(camel_permutations), "first 5 permutations:", camel_permutations[:5], "should permutations")

    ## include white camel as well as black camel:
    if 6 in camels_not_thrown:
        camels_not_thrown.remove(6)
        camels_not_thrown.append(7)
        all_camel_permutations = np.concatenate((camel_permutations, _all_camel_permutations(np.array(camels_not_thrown, dtype=np.int64))), axis=0)
        camels_not_thrown.remove(7)
        camels_not_thrown.append(6)
    else:
        all_camel_permutations = camel_permutations

    if verbose:
        print("Number of full permutations: ", all_camel_permutations.shape[0]*all_dice_permutations.shape[0])

    camel_row_idx_base = np.zeros(8,dtype=np.int64)
    camel_col_idx_base = np.zeros(8,dtype=np.int64)
    if 6 in camels_not_thrown:
        for i in range(1,8):
            pos = np.argwhere(rendered_field == i)
            camel_row_idx_base[i] = pos[0][0]
            camel_col_idx_base[i] = pos[0][1]
    else:
        for i in range(1,6):
            pos = np.argwhere(rendered_field == i)
            camel_row_idx_base[i] = pos[0][0]
            camel_col_idx_base[i] = pos[0][1]

    n_perm = len(all_camel_permutations)

    positions_local = np.zeros((n_perm, 5, 3), dtype=np.int64)
    DO_hits_local   = np.zeros((n_perm, n_players, 1), dtype=np.int64)

    ## path simulation
    for camel_order_idx in nb.prange(len(all_camel_permutations)):
        camel_order = all_camel_permutations[camel_order_idx]
        for dice_rolls in all_dice_permutations:
            ##create field copy
            field_copy = rendered_field.copy()

            camel_row_idx = camel_row_idx_base.copy()
            camel_col_idx = camel_col_idx_base.copy()

            for move_idx in range(len(dice_rolls)):
                move = dice_rolls[move_idx]
                camel = camel_order[move_idx]
                ## find camel in field_copy
                row = camel_row_idx[camel]
                col = camel_col_idx[camel]
                # Extract the stack as a vector (all nonzero from (row, col) to (row, end))
                # Optimized: slice out directly using numpy for speed (all leading nonzero, so from col to first 0 or end)
                # Directly operate on the field_copy row to avoid extra reference
                end = (field_copy[row] != 0).sum()

                stack = field_copy[row, col:end].copy().reshape(1, -1)
                # Zero out in-place
                field_copy[row, col:end] = 0

                ## move camel
                below_target = False
                if camel not in [6,7]:
                    target_field = row + move
                    if field_copy[target_field,0] == 8:
                        DO_hits_local[camel_order_idx, field_copy[target_field,1]-10, 0] += 1
                        target_field -= 1
                        below_target = True
                    elif field_copy[target_field,0] == 9:
                        DO_hits_local[camel_order_idx, field_copy[target_field,1]-10, 0] += 1
                        target_field += 1
                else:
                    target_field = row - move
                    if field_copy[target_field,0] == 8:
                        DO_hits_local[camel_order_idx, field_copy[target_field,1]-10, 0] += 1
                        target_field += 1
                        below_target = True
                    elif field_copy[target_field,0] == 9:
                        DO_hits_local[camel_order_idx, field_copy[target_field,1]-10, 0] += 1
                        target_field -= 1

                ##place camel on or below target
                #print(stack,field_copy[target_field,:-stack.shape[1]].reshape(1,-1))
                if below_target:
                    #field_copy[target_field, :] = np.concatenate((stack, field_copy[target_field, :-stack.shape[1]].reshape(1, -1)), axis=1)
                    stack_len = stack.shape[1]

                    # shift existing camels right
                    for j in range(6, stack_len - 1, -1):
                        field_copy[target_field, j] = field_copy[target_field, j - stack_len]

                    # insert stack at front
                    for j in range(stack_len):
                        field_copy[target_field, j] = stack[0, j]
                else:
                    end_target_field = (field_copy[target_field] != 0).sum()
                    field_copy[target_field,end_target_field:end_target_field+stack.shape[1]] = stack

                for i in range(7):
                    if field_copy[target_field,i] == 0:
                        break
                    else:
                        camel = field_copy[target_field,i]
                        camel_row_idx[camel] = target_field
                        camel_col_idx[camel] = i                        

                if target_field > 16:
                    break
            camel_positions = np.zeros(5)
            ## evaluation of the path
            for camel in range(1,6):
                row = camel_row_idx[camel]
                col = camel_col_idx[camel]
                camel_positions[camel-1] = row*7+col
            if camel_order_idx == 0 and verbose: print(camel_positions, camel_order, dice_rolls)
            camel_ranking = np.argsort(-camel_positions.flatten())
            positions_local[camel_order_idx, camel_ranking[0], 0] += 1
            positions_local[camel_order_idx, camel_ranking[1], 1] += 1
            for i in range(2,5):
                positions_local[camel_order_idx, camel_ranking[i], 2] += 1

    positions = positions_local.sum(axis=0)
    DO_hits   = DO_hits_local.sum(axis=0)
    return positions, DO_hits



class player(): #for simulation
    def __init__(self,name):
        while name in CamelUp.standard_Camels:
            print("ILLEGAL PLAYER NAME! Can't be Camel Name! Retry:")
            name = input()
        self.cards = copy.deepcopy(CamelUp.standard_Camels)
        self.name = name
        self.coins = 3
        self.inventory = []
        self.expected_payoff = 0
        self.plate_pos = None
    def end_of_round(self):
        self.coins += self.expected_payoff
        print_hint2(self.name + " earned " + str(int(self.expected_payoff)) +" this turn.")
        self.inventory = []
        self.expected_payoff = 0
        self.plate_pos = None




class Field():
    Camels = CamelUp.standard_Camels + ["Black", "White"]
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





