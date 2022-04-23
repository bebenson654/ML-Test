with open("SMSSpamCollection.txt", "r") as srcFile:
    lCount = 0  # line count
    hCount = 0  # ham count for balanced dataset
    sCount = 0  # spam count for balanced dataset

    bTestDir = ".\\Balanced\\Test\\"
    bTrainDir = ".\\Balanced\\Train\\"
    uTestDir = ".\\Unbalanced\\Test\\"
    uTrainDir = ".\\Unbalanced\\Train\\"

    for line in srcFile:
        line = line.strip().split("\t")

        if lCount <= 2787:
            f = open(
                uTrainDir + line[0].lower() + "\\" + "train_" + str(lCount) + ".txt",
                "w",
            )
            f.write(line[1])

        else:
            f = open(
                uTestDir + line[0].lower() + "\\" + "test_" + str(lCount) + ".txt",
                "w",
            )
            f.write(line[1])

        if hCount <= 373 and line[0].lower() == "ham":
            f = open(bTrainDir + "ham\\" + "train_" + str(lCount) + ".txt", "w")
            f.write(line[1])
            hCount += 1
        elif sCount <= 373 and line[0].lower() == "spam":
            f = open(bTrainDir + "spam\\" + "train_" + str(lCount) + ".txt", "w")
            f.write(line[1])
            sCount += 1
        else:
            f = open(
                bTestDir + line[0].lower() + "\\" + "test_" + str(lCount) + ".txt", "w"
            )
            f.write(line[1])

        lCount += 1


# 5574 total
# 2787 half
# 1393 quarter
# 747 spam
#   727/2 = 373.5
# 4827 ham
