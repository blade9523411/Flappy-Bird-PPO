import pygame
import sys
from game import Game

def main():
    pygame.init()
    clock = pygame.time.Clock()
    
    # Initialize the game
    game = Game()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.bird.flap()
        
        game.update()
        game.draw()
        
        clock.tick(60)  # Limit to 60 frames per second

if __name__ == "__main__":
    main()