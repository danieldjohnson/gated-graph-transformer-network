import json
import collections

Edge = collections.namedtuple("Edge",["source","dest","type"])

class GraphHelper( object ):
    def __init__(self):
        self.counters = collections.defaultdict(lambda: 0)
        self.nodes = set()
        self.edges = set()

    def dumps(self):
        return json.dumps({
            "nodes": sorted(self.nodes),
            "edges": [{"from":e.source,"to":e.dest,"type":e.type} for e in self.edges]
        })

    def make(self, node_type):
        full_name = node_type + "#" + str(self.counters[node_type])
        self.counters[node_type] += 1
        self.nodes.add(full_name)
        return Node(full_name, self)

    def make_unique(self, node_name):
        if not node_name in self.nodes:
            self.nodes.add(node_name)
        return Node(node_name, self)

class BadEdgeError( Exception ):
    pass

class Node( object ):
    def __init__(self, identifier, parent):
        object.__setattr__(self, 'identifier', identifier)
        object.__setattr__(self, 'parent', parent)

    def __getattr__(self, edgename):
        matching = set(e.dest for e in self.parent.edges
                        if e.source == self.identifier
                           and e.type == edgename)
        if len(matching) == 0:
            return None
        elif len(matching) > 1:
            raise BadEdgeError("Expected one result for {}.{}, got {}".format(self.identifier, edgename, matching))
        return Node(matching.pop(), self.parent)

    def __getitem__(self, edgename):
        return self.__getattr__(edgename)

    def __setattr__(self, edgename, value):
        if edgename in ["identifier","parent"]:
            self.__setattribute__(edgename, value)
            return
        matching = set(e for e in self.parent.edges
                        if e.source == self.identifier
                           and e.type == edgename)
        if len(matching) > 1:
            print("WARNING: Setting attr {} on {} clears old values, but has multiple edges {}".format(edgename,self.identifier,matching))
        self.parent.edges -= matching
        if value is not None:
            self.parent.edges.add(Edge(self.identifier, value.identifier, edgename))

    def __setitem__(self, edgename, value):
        return self.__setattr__(edgename, value)

    def getall(self, edgename):
        matching = frozenset(e.dest for e in self.parent.edges
                        if e.source == self.identifier
                           and e.type == edgename)
        return matching

    def add(self, edgename, dest):
        self.parent.edges.add(Edge(self.identifier, dest.identifier, edgename))

    def remove(self, edgename=None, dest=None):
        matching = set(e for e in self.parent.edges
                        if e.source == self.identifier
                           and (e.dest == dest.identifier or dest is None)
                           and (e.type == edgename or edgename is None))
        self.parent.edges -= matching

    @property
    def type(self):
        return self.identifier.split("#")[0]
    

class Story( object ):
    def __init__(self):
        self.graph = GraphHelper()
        self.counter = 1
        self.lines = []

    def add_line(self, line_str):
        assert not "=" in line_str
        assert not "\t" in line_str
        self.lines.append("{} {}={}".format(self.counter, line_str, self.graph.dumps()))
        self.counter += 1

    def no_query(self):
        self.add_query("","")

    def add_query(self, query, answer):
        assert not "=" in query + answer
        assert not "\t" in query + answer
        self.lines.append("{} {}\t{}".format(self.counter, query, answer))
        self.counter += 1






