
 
def print_adj(text,length,option="l",fill=" "):
    text =str(text)
    text_length = len(text)
    length_dif = length-text_length
    if length_dif<0:
        print("Error: text to long")
        return text
    if option == "l":
        return text+fill*length_dif
    elif option == "r":
        return " "*length_dif+text
    elif option == "c":
        return fill*int(length_dif/2)+text+" "*int(round(length_dif/2+0.499,2))
    elif option == "cl":
        return fill*int(round(length_dif/2+0.499,2))+text+" "*int(length_dif/2)
    else:
        print("unknown option!")
        return text

def print_header(text, factor= 1, space = 2, width = None, sign = "#", aspect_ratio = 16/9,vspace=1):
    '''
    
    '''
    height= (1+2*factor)*2
    width_internal = (len(text)+2*space+2*factor)
    adjust = 0
    if width is None:
        if width_internal/height<aspect_ratio:
            adjust = int((aspect_ratio/(width_internal/height)*width_internal-width_internal+2)/2)
            
        print(("\n"*vspace+sign*(len(text)+2*(factor+space+adjust)))*factor+"\n"+sign*(factor+adjust),text,
              sign*(factor+adjust)+("\n"+sign*(len(text)+2*(factor+space+adjust)))*factor+"\n"*vspace,sep=" "*space)
    else:
        width_field_min = space*2+factor*2+len(text)
        if width_field_min > width_internal:
            print(f"print_fmt: Minimum width too low! Is {width} and should be {width_field_min}!")
            return
        padding = width-len(text)-2*space
        padding_r = padding//2
        padding_l = padding-padding_r
        print("\n"*vspace+(width*sign+"\n")*factor+padding_l*sign+space*" "+text+\
              space*" "+padding_r*sign+("\n"+width*sign)*factor+"\n"*vspace) 