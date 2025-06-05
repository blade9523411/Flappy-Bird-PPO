import pygame

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Flappy Bird")
        self.bird = Bird()
        self.pipes = []
        self.pipe_interval = 90  # frames between new pipe pairs
        self.frame_count = 0
        self.game_over = False
        self.score = 0  # score attribute

    def update(self):
        if self.game_over:
            return

        self.bird.update()

        # Check boundaries: if the bird goes above or below, game over.
        if self.bird.y - self.bird.radius < 0 or self.bird.y + self.bird.radius > 600:
            self.game_over = True

        # Update pipes and remove those off-screen.
        for pipe in self.pipes:
            pipe.update()
        self.pipes = [pipe for pipe in self.pipes if not pipe.off_screen()]

        # Spawn a new pipe every pipe_interval frames with fixed parameters.
        if self.frame_count % self.pipe_interval == 0:
            # Fixed vertical position: every pipe has top_height 200 and gap_height 150.
            self.pipes.append(Pipe(400, top_height=200, gap_height=150))
        self.frame_count += 1

        # Check collisions and update score.
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + pipe.width < self.bird.x:
                self.score += 1
                pipe.scored = True
            if pipe.collides(self.bird):
                self.game_over = True

    def draw(self):
        self.screen.fill((135, 206, 235))  # sky blue background
        self.bird.draw(self.screen)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        
        # Draw boundary lines (red lines for top and bottom).
        pygame.draw.line(self.screen, (255, 0, 0), (0, 0), (400, 0), 5)
        pygame.draw.line(self.screen, (255, 0, 0), (0, 600), (400, 600), 5)
        
        # Draw the score.
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

class Bird:
    def __init__(self):
        self.x = 50
        self.y = 300
        self.velocity = 0
        self.gravity = 0.5
        self.radius = 15

    def flap(self):
        self.velocity = -10

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self, x, top_height=200, gap_height=150):
        self.x = x
        self.width = 50
        self.speed = 3  # Pipes move left at a constant speed.
        self.gap_height = gap_height
        self.top_height = top_height
        self.bottom_y = self.top_height + self.gap_height
        self.scored = False

    def update(self):
        self.x -= self.speed

    def off_screen(self):
        return self.x + self.width < 0

    def draw(self, screen):
        # Draw top pipe.
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(self.x, 0, self.width, self.top_height))
        # Draw bottom pipe.
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(self.x, self.bottom_y, self.width, 600 - self.bottom_y))

    def collides(self, bird):
        bird_rect = bird.get_rect()
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_pipe_rect = pygame.Rect(self.x, self.bottom_y, self.width, 600 - self.bottom_y)
        return bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect)
    