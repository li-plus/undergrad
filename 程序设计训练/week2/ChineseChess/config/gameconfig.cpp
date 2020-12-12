#include "gameconfig.h"


GameConfig gameConfig;

GameConfig::GameConfig()
{
    this->stepSec = 30;
    this->side = RED;
    this->appType = SERVER;
    this->showHint = true;
    this->isAudioOn = true;
}

void GameConfig::setGameConfig(GameSide side, AppType appType, bool showHint, int stepSec, bool isAudioOn)
{
    this->side = side;
    this->appType = appType;
    this->showHint = showHint;
    this->stepSec = stepSec;
    this->isAudioOn = isAudioOn;
}
