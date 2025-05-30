function Game() {
    this.map = [];
    this.width = 40;
    this.height = 24;
    this.hero = null;
    this.enemies = [];
    this.items = [];
    this.rooms = [];
    this.keys = {}; 
    
    this.TILE_TYPES = {
        WALL: 'W',
        EMPTY: '-',
        HERO: 'P',
        ENEMY: 'E',
        SWORD: 'SW',
        HEALTH_POTION: 'HP'
    };
    
    this.init = function() {
        this.generateMap();
        this.placeRooms();
        this.connectRooms();
        this.placeItems();
        this.placeHero();
        this.placeEnemies();
        this.render();
        this.setupControls();
        this.startGameLoop();
    };
    
    this.generateMap = function() {
        this.map = [];
        for (let y = 0; y < this.height; y++) {
            let row = [];
            for (let x = 0; x < this.width; x++) {
                row.push(this.TILE_TYPES.WALL);
            }
            this.map.push(row);
        }
    };
    
    this.placeRooms = function() {
        const numRooms = Math.floor(Math.random() * 6) + 5; 
        this.rooms = [];
        
        for (let i = 0; i < numRooms; i++) {
            const width = Math.floor(Math.random() * 6) + 3; 
            const height = Math.floor(Math.random() * 6) + 3; 
            const x = Math.floor(Math.random() * (this.width - width - 2)) + 1;
            const y = Math.floor(Math.random() * (this.height - height - 2)) + 1;
            
            let overlaps = false;
            for (let room of this.rooms) {
                if (x < room.x + room.width + 1 && x + width + 1 > room.x &&
                    y < room.y + room.height + 1 && y + height + 1 > room.y) {
                    overlaps = true;
                    break;
                }
            }
            
            if (!overlaps) {
                for (let dy = 0; dy < height; dy++) {
                    for (let dx = 0; dx < width; dx++) {
                        this.map[y + dy][x + dx] = this.TILE_TYPES.EMPTY;
                    }
                }
                
                this.rooms.push({
                    x: x,
                    y: y,
                    width: width,
                    height: height,
                    centerX: Math.floor(x + width / 2),
                    centerY: Math.floor(y + height / 2)
                });
            } else {
                i--; 
            }
        }
    };
    
    this.connectRooms = function() {
        this.rooms.sort((a, b) => a.centerX - b.centerX);
        
        for (let i = 0; i < this.rooms.length - 1; i++) {
            const room1 = this.rooms[i];
            const room2 = this.rooms[i + 1];
            
            const startX = room1.centerX;
            const startY = room1.centerY;
            const endX = room2.centerX;
            const endY = room2.centerY;
            
            const x1 = Math.min(startX, endX);
            const x2 = Math.max(startX, endX);
            for (let x = x1; x <= x2; x++) {
                this.map[startY][x] = this.TILE_TYPES.EMPTY;
            }
            
            const y1 = Math.min(startY, endY);
            const y2 = Math.max(startY, endY);
            for (let y = y1; y <= y2; y++) {
                this.map[y][endX] = this.TILE_TYPES.EMPTY;
            }
        }
        
        const numExtraCorridors = Math.floor(Math.random() * 3) + 2; 
        for (let i = 0; i < numExtraCorridors; i++) {
            const room1 = this.rooms[Math.floor(Math.random() * this.rooms.length)];
            const room2 = this.rooms[Math.floor(Math.random() * this.rooms.length)];
            
            if (room1 !== room2) {
                const startX = room1.centerX;
                const startY = room1.centerY;
                const endX = room2.centerX;
                const endY = room2.centerY;
                
                const x1 = Math.min(startX, endX);
                const x2 = Math.max(startX, endX);
                for (let x = x1; x <= x2; x++) {
                    this.map[startY][x] = this.TILE_TYPES.EMPTY;
                }
                
                const y1 = Math.min(startY, endY);
                const y2 = Math.max(startY, endY);
                for (let y = y1; y <= y2; y++) {
                    this.map[y][endX] = this.TILE_TYPES.EMPTY;
                }
            }
        }
    };
    
    this.placeItems = function() {
        for (let i = 0; i < 2; i++) {
            this.placeRandomItem(this.TILE_TYPES.SWORD);
        }
        
        for (let i = 0; i < 10; i++) {
            this.placeRandomItem(this.TILE_TYPES.HEALTH_POTION);
        }
    };
    
    this.placeRandomItem = function(itemType) {
        let placed = false;
        let attempts = 0;
        while (!placed && attempts < 100) {
            const x = Math.floor(Math.random() * this.width);
            const y = Math.floor(Math.random() * this.height);
            
            if (this.map[y][x] === this.TILE_TYPES.EMPTY) {
                this.map[y][x] = itemType;
                this.items.push({x: x, y: y, type: itemType});
                placed = true;
            }
            attempts++;
        }
    };
    
    this.placeHero = function() {
        let placed = false;
        while (!placed) {
            const x = Math.floor(Math.random() * this.width);
            const y = Math.floor(Math.random() * this.height);
            
            if (this.map[y][x] === this.TILE_TYPES.EMPTY) {
                this.hero = {
                    x: x,
                    y: y,
                    health: 100,
                    attack: 10,
                    hasSword: false
                };
                this.map[y][x] = this.TILE_TYPES.HERO;
                placed = true;
            }
        }
    };
    
    this.placeEnemies = function() {
        for (let i = 0; i < 10; i++) {
            let placed = false;
            while (!placed) {
                const x = Math.floor(Math.random() * this.width);
                const y = Math.floor(Math.random() * this.height);
                
                if (this.map[y][x] === this.TILE_TYPES.EMPTY) {
                    this.enemies.push({
                        x: x,
                        y: y,
                        health: 50,
                        attack: 15
                    });
                    this.map[y][x] = this.TILE_TYPES.ENEMY;
                    placed = true;
                }
            }
        }
    };
    
    this.render = function() {
        const field = $('.field');
        field.empty();
        
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const tile = $('<div>').addClass('tile');
                
                switch (this.map[y][x]) {
                    case this.TILE_TYPES.WALL:
                        tile.addClass('tileW');
                        break;
                    case this.TILE_TYPES.HERO:
                        tile.addClass('tileP');
                        if (this.hero) {
                            const health = $('<div>').addClass('health')
                                .css('width', this.hero.health + '%');
                            tile.append(health);
                        }
                        break;
                    case this.TILE_TYPES.ENEMY:
                        tile.addClass('tileE');
                        const enemy = this.enemies.find(e => e.x === x && e.y === y);
                        if (enemy) {
                            const health = $('<div>').addClass('health')
                                .css('width', (enemy.health / 50) * 100 + '%');
                            tile.append(health);
                        }
                        break;
                    case this.TILE_TYPES.SWORD:
                        tile.addClass('tileSW');
                        break;
                    case this.TILE_TYPES.HEALTH_POTION:
                        tile.addClass('tileHP');
                        break;
                }
                
                tile.css({
                    left: x * 30 + 'px',
                    top: y * 30 + 'px'
                });
                
                field.append(tile);
            }
        }
    };
    
    this.setupControls = function() {
        var self = this;
        $(document).on('keydown', function(e) {
            e.preventDefault();
            const key = e.key.toLowerCase();
            
            switch(key) {
                case 'w':
                case 'ц':
                    self.moveHero(0, -1);
                    break;
                case 's':
                case 'ы':
                    self.moveHero(0, 1);
                    break;
                case 'a':
                case 'ф':
                    self.moveHero(-1, 0);
                    break;
                case 'd':
                case 'в':
                    self.moveHero(1, 0);
                    break;
                case ' ':
                    self.heroAttack();
                    break;
            }
        });
    };
    
    this.startGameLoop = function() {
        var self = this;
        setInterval(function() {
            self.gameLoop();
        }, 100);
    };
    
    this.gameLoop = function() {
        if (this.keys['w'] || this.keys['ц']) {
            this.moveHero(0, -1);
        }
        if (this.keys['s'] || this.keys['ы']) {
            this.moveHero(0, 1);
        }
        if (this.keys['a'] || this.keys['ф']) {
            this.moveHero(-1, 0);
        }
        if (this.keys['d'] || this.keys['в']) {
            this.moveHero(1, 0);
        }
        if (this.keys[' ']) {
            this.heroAttack();
        }
    };
    
    this.moveEnemies = function() {
        for (let enemy of this.enemies) {
            const directions = [
                {dx: 0, dy: -1}, 
                {dx: 0, dy: 1},  
                {dx: -1, dy: 0}, 
                {dx: 1, dy: 0}   
            ];
            
            const dir = directions[Math.floor(Math.random() * directions.length)];
            const newX = enemy.x + dir.dx;
            const newY = enemy.y + dir.dy;
            
            if (newX >= 0 && newX < this.width && newY >= 0 && newY < this.height) {
                if (this.map[newY][newX] === this.TILE_TYPES.EMPTY) {
                    this.map[enemy.y][enemy.x] = this.TILE_TYPES.EMPTY;
                    enemy.x = newX;
                    enemy.y = newY;
                    this.map[newY][newX] = this.TILE_TYPES.ENEMY;
                }
            }
        }
    };
    
    this.checkEnemyAttacks = function() {
        const directions = [
            {dx: 0, dy: -1}, 
            {dx: 0, dy: 1},  
            {dx: -1, dy: 0}, 
            {dx: 1, dy: 0}  
        ];
        
        for (let dir of directions) {
            const checkX = this.hero.x + dir.dx;
            const checkY = this.hero.y + dir.dy;
            
            if (checkX >= 0 && checkX < this.width && checkY >= 0 && checkY < this.height) {
                if (this.map[checkY][checkX] === this.TILE_TYPES.ENEMY) {
                    const enemy = this.enemies.find(e => e.x === checkX && e.y === checkY);
                    if (enemy) {
                        this.hero.health -= enemy.attack;
                    }
                }
            }
        }
    };
    
    this.moveHero = function(dx, dy) {
        const newX = this.hero.x + dx;
        const newY = this.hero.y + dy;
        
        if (newX >= 0 && newX < this.width && newY >= 0 && newY < this.height) {
            const targetTile = this.map[newY][newX];
            
            if (targetTile === this.TILE_TYPES.EMPTY || 
                targetTile === this.TILE_TYPES.SWORD || 
                targetTile === this.TILE_TYPES.HEALTH_POTION) {
                
                this.map[this.hero.y][this.hero.x] = this.TILE_TYPES.EMPTY;
                
                if (targetTile === this.TILE_TYPES.SWORD) {
                    this.hero.hasSword = true;
                    this.items = this.items.filter(item => !(item.x === newX && item.y === newY));
                } else if (targetTile === this.TILE_TYPES.HEALTH_POTION) {
                    const maxHealth = 100;
                    const healAmount = Math.floor(maxHealth * 0.5);
                    this.hero.health = Math.min(maxHealth, this.hero.health + healAmount);
                    this.items = this.items.filter(item => !(item.x === newX && item.y === newY));
                }
                
                this.hero.x = newX;
                this.hero.y = newY;
                this.map[newY][newX] = this.TILE_TYPES.HERO;
                
                this.moveEnemies();
                
                this.checkEnemyAttacks();
                
                this.render();
                
                if (this.hero.health <= 0) {
                    this.render();
                    alert('Game Over!');
                    location.reload();
                }
            }
        }
    };
    
    this.heroAttack = function() {
        const directions = [
            {dx: 0, dy: -1},
            {dx: 0, dy: 1},
            {dx: -1, dy: 0},
            {dx: 1, dy: 0}
        ];
        
        let attackUsed = false;
        
        for (let dir of directions) {
            const newX = this.hero.x + dir.dx;
            const newY = this.hero.y + dir.dy;
            
            if (newX >= 0 && newX < this.width && newY >= 0 && newY < this.height) {
                if (this.map[newY][newX] === this.TILE_TYPES.ENEMY) {
                    const enemy = this.enemies.find(e => e.x === newX && e.y === newY);
                    if (enemy) {
                        const damage = this.hero.hasSword && !attackUsed ? this.hero.attack * 3 : this.hero.attack;
                        enemy.health -= damage;
                        this.hero.health -= enemy.attack;
                        attackUsed = true;
                        
                        if (enemy.health <= 0) {
                            this.map[newY][newX] = this.TILE_TYPES.EMPTY;
                            this.enemies = this.enemies.filter(e => !(e.x === newX && e.y === newY));
                        }
                    }
                }
            }
        }
        
        if (attackUsed) {
            this.hero.hasSword = false;
        }
        
        this.render();
        
        if (this.hero.health <= 0) {
            this.render();
            alert('Game Over!');
            location.reload();
        }
    };
} 