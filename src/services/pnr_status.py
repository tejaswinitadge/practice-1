
def get_pnr_status(pnr):
    if(pnr):
        print("User came with pnr ", pnr)
    # db check

    print("User pw check failed with pnr ", pnr)

    if pnr=="12345":
        return "confirmed"
    return "waiting"