#ifndef GAMECONFIG_H
#define GAMECONFIG_H


#include "definition.h"


class GameConfig
{
public:
    GameConfig();

    // writer
    void setGameConfig(GameSide side, AppType appType, bool showHint, int stepSec, bool isAudioOn);
    void setGameSide(GameSide side) { this->side = side; }
    void setAppType(AppType type){ this->appType = type; }
    void setShowHint(bool hint){ this->showHint = hint; }
    void setStepSec(int sec){ this->stepSec = sec; }
    void setAudioOn(bool isAudioOn) { this->isAudioOn = isAudioOn; }
    // reader
    AppType getAppType() const {return this->appType; }
    GameSide getMySide() const { return this->side; }
    GameSide getOtherSide() const { return this->side==BLACK ? RED : BLACK; }
    bool getShowHint() const { return this->showHint; }
    int getStepSec() const { return this->stepSec; }
    bool getAudioOn() const { return this->isAudioOn; }

    // helper
    static GameSide getOppositeSide(GameSide side){ return side == RED ? BLACK : RED; }
private:
    GameSide side;
    AppType appType;
    bool showHint;
    int stepSec;
    bool isAudioOn;

};

extern GameConfig gameConfig;

#endif // GAMECONFIG_H
