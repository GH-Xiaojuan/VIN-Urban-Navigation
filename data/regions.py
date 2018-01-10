from collections import defaultdict

class RegionPartitioner(object):
    def __init__(self, M, N, start_longitude, end_longitude, start_latitude, end_latitude):
        self.M = M
        self.N = N
        self.start_longitude = start_longitude
        self.end_longitude = end_longitude
        self.start_latitude = start_latitude
        self.end_latitude = end_latitude

        width = end_longitude - start_longitude
        height = end_latitude - start_latitude
        self.x_step = height / M
        self.y_step = width / N

    def generate_region_map(self):
        """
        generate region number of each partitioned region.
        """
        M, N = self.M, self.N
        map = [['' for _ in range(N + 2)] for _ in range(M + 2)]
        map[0][0] = 'NW'
        map[M + 1][0] = 'SW'
        map[0][N + 1] = 'NE'
        map[M + 1][N + 1] = 'SE'
        for i in range(0, N):
            map[0][i + 1] = 'N' + str(i)
            map[M + 1][i + 1] = 'S' + str(i)
        for i in range(0, M):
            map[i + 1][0] = 'W' + str(i)
            map[i + 1][N + 1] = 'E' + str(i)

        for i in range(0, M):
            for j in range(0, N):
                map[i + 1][j + 1] = str(i * N + j)

        return map


    def generate_region_dict(self, region_map):
        region_dict = defaultdict(tuple)
        for i in range(self.M + 2):
            for j in range(self.N + 2):
                region_dict[region_map[i][j]] = (i, j)
        return region_dict


    def adjacents(self, region_map, region_dict):
        """
        @:return a adjacent map indicating whether two regions are adjacent.
        """

        def get_direction(cur, next):
            """
            (N, NE, E, SE, S, SW, W, NW)
            (0, 1, 2, 3, 4, 5, 6, 7)
            dirs = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
            """
            dirs = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
            cur_i, cur_j = region_dict[cur]
            next_i, next_j = region_dict[next]
            for i in range(len(dirs)):
                if cur_i + dirs[i][0] == next_i and cur_j + dirs[i][1] == next_j:
                    return i + 1
            return -1

        adjacents = dict()
        regions = [r for row in region_map for r in row]
        for prev in regions:
            for cur in regions:
                adjacents[prev + '_' + cur] = get_direction(prev, cur)
        return adjacents


    def partition_regions(self):
        region_map = self.generate_region_map()
        region_dict = self.generate_region_dict(region_map)
        adjs = self.adjacents(region_map, region_dict)
        return region_map, region_dict, adjs


    def get_region(self, longitude, latitude, ):
        """
        Get the region number of [longitude, latitude]
        """
        start_latitude, end_latitude = self.start_latitude, self.end_latitude
        start_longitude, end_longitude = self.start_longitude, self.end_longitude
        x_step, y_step = self.x_step, self.y_step
        M, N = self.M, self.N

        def get_latitude_number():
            return N - int((latitude - start_latitude) // x_step) - 1

        def get_longitude_number():
            return int((longitude - start_longitude) // y_step)

        if longitude < start_longitude:
            if latitude < start_latitude:
                return 'SW'
            elif latitude >= end_latitude:
                return 'NW'
            else:
                return 'W' + str(get_latitude_number())
        elif longitude > end_longitude:
            if latitude < start_latitude:
                return 'SE'
            elif latitude >= end_latitude:
                return 'NE'
            else:
                return 'E' + str(get_latitude_number())
        else:
            if latitude < start_latitude:
                return 'S' + str(get_longitude_number())
            elif latitude >= end_latitude:
                return 'N' + str(get_longitude_number())
            else:
                return str(get_latitude_number() * N +
                           get_longitude_number())