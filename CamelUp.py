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
            5 Camels (White, Blue, Orange, Green, Yellow)
        Inventory:
            Betting Plates for each of the Camels
    '''
    gap_margin = 5
    Camels = ["Yellow","Blue","Green","Orange","White"]
    Inventory = [] # contains 
    for i in Camels:
        Inventory.extend([i+" [5]",i+" [3]",i+" [2]"])
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
    render_field_cell_width = 12
    parallel_workers= 8
    def __init__(self,
                 n_players:int,
                 start:bool = True,
                 field = "",
                 tutorial=True):
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
                    first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
                    second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
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
        first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
        second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
        # determine camels that haven't moved:
        Camels_die = copy.deepcopy(self.Camels)
        for i in self.moved:
            if i in Camels_die:
                del Camels_die[Camels_die.index(i)]
        # determine current field of play
        start_field = copy.deepcopy(self.game_field)
        payoff = {}
        for i in range(len(start_field)):
            if "OASIS" in start_field[i] or "DESERT" in start_field[i]:
                payoff[str(i+1)] = 0
        ## payoffs and win probabilities might be available already from previous calculation.
        if self.rec:
            first,second,payoff,n_paths = \
            self.flexible_for(liste = Camels_die, #list of camels that haven't moved
                              field= start_field, # current field of play
                              first= first, # 
                              second= second, #
                              payoff= payoff) #
            self.game_first = first; self.game_second = second;
            self.game_payoff = payoff; self.game_n_paths = n_paths;
        else:
            first,second,payoff,n_paths = \
                self.game_first, self.game_second, self.game_payoff, self.game_n_paths
        #payoff:
        self.win_prob = {}
        self.sec_prob = {}
        self.lose_prob = {}
        for i in first.keys():
            self.win_prob[i]=first[i]/n_paths
            self.sec_prob[i]=second[i]/n_paths
            self.lose_prob[i] = 1-self.win_prob[i]-self.sec_prob[i]
        self.ret = {}
        for i in self.win_prob.keys():
            self.ret[i+" [5]"]=5*self.win_prob[i]+self.sec_prob[i]-self.lose_prob[i]
            self.ret[i+" [3]"]=3*self.win_prob[i]+self.sec_prob[i]-self.lose_prob[i]
            self.ret[i+" [2]"]=2*self.win_prob[i]+self.sec_prob[i]-self.lose_prob[i]
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
        first = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
        second = {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0}
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
                if field[i] == ["black"] or field[i] == ["white"] and not camel_found:
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
    # @numba.njit # currently won't work. rewrite without using dictionaries
    def new_flexible_for(self,moves,camels,field,payoffs):
        '''
        

        Parameters
        ----------
        moves : TYPE
            DESCRIPTION.
        camels : TYPE
            DESCRIPTION.
        field : TYPE
            DESCRIPTION.
        payoffs : TYPE
            DESCRIPTION.

        Returns
        -------
        results : TYPE
            DESCRIPTION.
        
        --------
        Testing:
        --------
            
            payoff = {} 
            for i in range(len(CCS2.game_field)):
                if "OASIS" in CCS2.game_field[i] or "DESERT" in CCS2.game_field[i]:
                    payoff[str(i+1)] = 0
        
            a_time = time.time()
            results = CCS2.new_flexible_for(
                [],
                [camel for camel in CCS2.Camels if camel not in CCS2.moved],
                CCS2.game_field + [],
                payoff
                )
            print(time.time()-a_time)
        '''
        # import pdb; pdb.set_trace()
        self.counter = 0
        self.counter_limit = 29160
        for i in range(int(len(moves)/2)):
            self.counter_limit/=(5-i)*3
        self.jobs = []
        self.results = []
        for camel in camels:
            camels_copy = copy.deepcopy(camels)
            camels_copy.remove(camel)
            self.jobs.append([moves,camel,camels_copy,field,payoffs])
        threads = []
        for worker in range(self.parallel_workers):
            threads.append(th.Thread(target=self.new_flexible_for_worker))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # results = pd.DataFrame(self.results,columns=[
        #     *[x for xs in [["Camel{:d}".format(i),"Move{:d}".format(i)] for i in range(1,6)] for x in xs],
        #     "weight","second","first",*payoffs.keys()])
        return self.results
    def new_flexible_for_worker(self):
        time_waited = 0
        while self.counter < self.counter_limit:
            if len(self.jobs)==0:
                time.sleep(.005)
                time_waited += .005
                if time_waited>=5:
                    return
                continue
            time_waited = 0
            moves,camel,camels,field,hits = self.jobs.pop()
            for j in range(1,4):
                moves_copy = copy.deepcopy(moves) +[camel,j]
                hits_copy = copy.deepcopy(hits)
                camels_copy = copy.deepcopy(camels)
                field1 = copy.deepcopy(field)
                field2,hit = self.move_simulation(field1, camel,j)
                if len(hit.values())>0:
                    key = list(hit.keys())[0]
                    hits_copy[key]+=1
                if len([*field2[16],*field2[17],*field2[18]]) > 0 or\
                    len(camels_copy) == 0:
                    ranks = self.rank(field2)
                    factor = np.prod([i*3 for i in range(1,1+len(camels_copy))])
                    self.counter+=factor
                    self.results.append([
                        *moves_copy,*["",0]*(len(camels_copy)),
                        factor,*ranks[-2:],*hits_copy.values()])
                else:
                    for camel2 in camels_copy:
                        camels_copy2 = camels_copy+[]
                        camels_copy2.remove(camel2)
                        self.jobs.append([moves_copy,camel2,camels_copy2,field2,hits])
                
    def flexible_for(self,liste,field,first,second,payoff):
        '''
        This function simulates the paths for the moves of the camels 

        Parameters
        ----------
        liste : list
            Camels that still move.
        field : list of lists
            Current field.
        first : dict of len 5
            DESCRIPTION.
        second : dict of len 5
            DESCRIPTION.
        payoff : dict
            DESCRIPTION.

        Returns
        -------
        first : TYPE
            See Parameters.
        second : TYPE
            See Parameters.
        payoff : TYPE
            See Parameters.
        n_paths: int
            See Parameters.
            
        --------
        Testing:
        --------
        
            payoff = {} 
            for i in range(len(CCS2.game_field)):
                if "OASIS" in CCS2.game_field[i] or "DESERT" in CCS2.game_field[i]:
                    payoff[str(i+1)] = 0
        
            a_time = time.time()
            first,second,payoff,npaths =   CCS2.flexible_for(
                [camel for camel in CCS2.Camels if camel not in CCS2.moved],
                CCS2.game_field + [],
                {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0},
                {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0},
                payoff)
            print(time.time()-a_time)
            
        '''
        if len(liste) > 1:
            npaths = 0
            for i in range(len(liste)):
                liste_2 = copy.deepcopy(liste)
                del liste_2[i]
                for j in range(1,4):
                    field1 = copy.deepcopy(field)
                    field2,hit = self.move_simulation(field1, liste[i],j)
                    if len([*field2[16],*field2[17],*field2[18]]) > 0:
                        ranks = self.rank(field2)
                        first[ranks[-1]]+=1*3**(len(liste)-1)
                        second[ranks[-2]]+=1*3**(len(liste)-1)
                        if len(hit.values())>0:
                            key = list(hit.keys())[0]
                            payoff[key]+=hit[key]
                        npaths += 1*3**(len(liste)-1)
                        continue
                    first,second,payoff,n_paths1 = self.flexible_for(liste_2,field2,first,second,payoff)
                    if len(hit.values())>0:
                        key = list(hit.keys())[0]
                        payoff[key]+=hit[key]*n_paths1
                    npaths += n_paths1
            return first,second,payoff,npaths
        elif len(liste) == 1:
            npaths = 0
            for i in range(1,4):
                field1 = copy.deepcopy(field)
                field2,hit = self.move_simulation(field1,liste[0],i)
                ranks = self.rank(field2)
                first[ranks[-1]]+=1
                second[ranks[-2]]+=1
                if len(hit.values())>0:
                    key = list(hit.keys())[0]
                    payoff[key]+=hit[key]
                npaths += 1
            return first,second,payoff,npaths
        else:
            ranks = self.rank(field)
            first[ranks[-1]]+=1
            second[ranks[-2]]+=1
            return first,second,payoff,1
    def flexible_for2(self, liste:list, field, first:dict, second:dict, n_paths:int):
        pass
    def sub_unit(self,liste,field,first,second,payoff,npaths):    
        i = self.sub_tasks.pop()
        liste_2 = copy.deepcopy(liste)
        del liste_2[i]
        for j in range(1,4):
            field1 = copy.deepcopy(field)
            field2,hit = self.move_simulation(field1, liste[i],j)
            if len([*field2[16],*field2[17],*field2[18]]) > 0:
                ranks = self.rank(field2)
                first[ranks[-1]]+=1*3**(len(liste)-1)
                second[ranks[-2]]+=1*3**(len(liste)-1)
                if len(hit.values())>0:
                    key = list(hit.keys())[0]
                    payoff[key]+=hit[key]
                npaths += 1*3**(len(liste)-1)
                continue
            first,second,payoff,n_paths1 = self.flexible_for(liste_2,field2,first,second,payoff)
            if len(hit.values())>0:
                key = list(hit.keys())[0]
                payoff[key]+=hit[key]*n_paths1
            npaths += n_paths1
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
    mask_players = {name: i for i, name in enumerate(Field.players.keys())}
    mask = {"White":0,"Blue":1,"Orange":2,"Yellow":3,"Green":4, "White":5, "Black":6} ## mask used to map field for numba acceleration
    for player_name, idx in mask_players.items():
        mask["DESERT"+player_name] = 7 + idx*2
        mask["OASIS"+player_name] = 8 + idx*2

    rendered_field = np.zeros((len(Field.game_field),7),dtype=int)
    for i in range(len(Field.game_field)):
        if len(Field.game_field[i]) > 0:
            if Field.game_field[i][0] in ["DESERT","OASIS"]:
                rendered_field[i,0] = mask[Field.game_field[i][0]+Field.game_field[i][1]]
            else:
                for j in range(len(Field.game_field[i])):
                    rendered_field[i,j] = mask[Field.game_field[i][j]]
    return rendered_field

@nb.njit()
def sim_all_moves(rendered_field:np.ndarray):
    #def flexible_for(self,liste,field,first,second,payoff):
        '''
        This function simulates the paths for the moves of the camels 

        Parameters
        ----------
        liste : list
            Camels that still move.
        field : list of lists
            Current field.
        first : dict of len 5
            DESCRIPTION.
        second : dict of len 5
            DESCRIPTION.
        payoff : dict
            DESCRIPTION.

        Returns
        -------
        first : TYPE
            See Parameters.
        second : TYPE
            See Parameters.
        payoff : TYPE
            See Parameters.
        n_paths: int
            See Parameters.
            
        --------
        Testing:
        --------
        
            payoff = {} 
            for i in range(len(CCS2.game_field)):
                if "OASIS" in CCS2.game_field[i] or "DESERT" in CCS2.game_field[i]:
                    payoff[str(i+1)] = 0
        
            a_time = time.time()
            first,second,payoff,npaths =   CCS2.flexible_for(
                [camel for camel in CCS2.Camels if camel not in CCS2.moved],
                CCS2.game_field + [],
                {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0},
                {"Yellow":0,"Blue":0,"Green":0,"Orange":0,"White":0},
                payoff)
            print(time.time()-a_time)
            
        '''
        if len(liste) > 1:
            npaths = 0
            for i in range(len(liste)):
                liste_2 = copy.deepcopy(liste)
                del liste_2[i]
                for j in range(1,4):
                    field1 = copy.deepcopy(field)
                    field2,hit = self.move_simulation(field1, liste[i],j)
                    if len([*field2[16],*field2[17],*field2[18]]) > 0:
                        ranks = self.rank(field2)
                        first[ranks[-1]]+=1*3**(len(liste)-1)
                        second[ranks[-2]]+=1*3**(len(liste)-1)
                        if len(hit.values())>0:
                            key = list(hit.keys())[0]
                            payoff[key]+=hit[key]
                        npaths += 1*3**(len(liste)-1)
                        continue
                    first,second,payoff,n_paths1 = self.flexible_for(liste_2,field2,first,second,payoff)
                    if len(hit.values())>0:
                        key = list(hit.keys())[0]
                        payoff[key]+=hit[key]*n_paths1
                    npaths += n_paths1
            return first,second,payoff,npaths
        elif len(liste) == 1:
            npaths = 0
            for i in range(1,4):
                field1 = copy.deepcopy(field)
                field2,hit = self.move_simulation(field1,liste[0],i)
                ranks = self.rank(field2)
                first[ranks[-1]]+=1
                second[ranks[-2]]+=1
                if len(hit.values())>0:
                    key = list(hit.keys())[0]
                    payoff[key]+=hit[key]
                npaths += 1
            return first,second,payoff,npaths
        else:
            ranks = self.rank(field)
            first[ranks[-1]]+=1
            second[ranks[-2]]+=1
            return first,second,payoff,1


class player(): #for simulation
    def __init__(self,name):
        while name in CamelUp.Camels:
            print("ILLEGAL PLAYER NAME! Can't be Camel Name! Retry:")
            name = input()
        self.cards = copy.deepcopy(CamelUp.Camels)
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
    Camels = CamelUp.Camels
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


print()





