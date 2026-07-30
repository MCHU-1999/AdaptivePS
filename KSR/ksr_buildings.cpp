#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Kinetic_surface_reconstruction_3.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>

using Kernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
using FT        = typename Kernel::FT;
using Point_3   = typename Kernel::Point_3;
using Vector_3  = typename Kernel::Vector_3;
using Segment_3 = typename Kernel::Segment_3;

using Point_set    = CGAL::Point_set_3<Point_3>;
using Point_map    = typename Point_set::Point_map;
using Normal_map   = typename Point_set::Vector_map;
 
using KSR = CGAL::Kinetic_surface_reconstruction_3<Kernel, Point_set, Point_map, Normal_map>;

int main(int argc, char** argv) {
  // Input and CLI args.
  std::string input_file, output_dir;
  auto print_usage = [&](const char* prog){
    std::cout << "Usage: " << prog << " [-i|--input <input.ply>] [-o|--output <output_dir>]" << std::endl;
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { print_usage(argv[0]); return EXIT_SUCCESS; }
    else if (a == "-i" || a == "--input") {
      if (i + 1 < argc) input_file = argv[++i]; else { std::cerr << "Error: missing argument for " << a << std::endl; print_usage(argv[0]); return EXIT_FAILURE; }
    } else if (a == "-o" || a == "--output") {
      if (i + 1 < argc) output_dir = argv[++i]; else { std::cerr << "Error: missing argument for " << a << std::endl; print_usage(argv[0]); return EXIT_FAILURE; }
    } else {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  namespace fs = std::filesystem;
  if (!fs::exists(input_file)) {
    std::cerr << "Input file not found: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  fs::path outdir(output_dir);
  std::error_code ec;
  if (!fs::exists(outdir)) {
    if (!fs::create_directories(outdir, ec)) {
      std::cerr << "Failed to create output directory '" << output_dir << "': " << ec.message() << std::endl;
      return EXIT_FAILURE;
    }
  } else if (!fs::is_directory(outdir)) {
    std::cerr << "Output path exists and is not a directory: " << output_dir << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading input: " << input_file << std::endl;
  std::cout << "Writing outputs to: " << outdir << std::endl;

  Point_set point_set;
  if (!CGAL::IO::read_point_set(input_file, point_set)) {
    std::cerr << "Failed to read point set from: " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  std::map<typename KSR::KSP::Face_support, bool> external_nodes;
  // All bbox faces prefer "outside" label except YMAX (intentional for model orientation).
  external_nodes[KSR::KSP::Face_support::ZMIN] = true;
  external_nodes[KSR::KSP::Face_support::ZMAX] = false;
  external_nodes[KSR::KSP::Face_support::XMIN] = false;
  external_nodes[KSR::KSP::Face_support::XMAX] = false;
  external_nodes[KSR::KSP::Face_support::YMIN] = false;
  external_nodes[KSR::KSP::Face_support::YMAX] = false;
 
  auto param =CGAL::parameters::k_neighbors(8)
    .maximum_distance(0.05) // the maximum distance from a point to a plane
    .maximum_angle(15)  // the maximum angle in degrees between the normal associated
                        // with a point and the normal of a plane
    .minimum_region_size(100) // the minimum number of points a region must have
    .regularize_parallelism(true) // whether parallelism should be regularized or not
    .regularize_coplanarity(true) // whether coplanarity should be regularized or not
    .regularize_orthogonality(true) // whether orthogonality should be regularized or not
    .angle_tolerance(15)   // Idk
    .maximum_offset(0.05); // maximum distance between two parallel planes
                            // to be considered coplanar

  // Algorithm.
  KSR ksr(point_set, param);
  ksr.detection_and_partition(2, param);

  std::vector<Point_3> vtx;
  std::vector<std::vector<std::size_t> > polylist;
  std::vector<FT> lambdas{0.1, 0.3, 0.5, 0.7, 0.9};
 

  bool save_biggest_component_only = true;
  bool non_empty = false;
  for (FT l : lambdas) {
    vtx.clear();
    polylist.clear();
    // ksr.reconstruct_with_ground(l, std::back_inserter(vtx), std::back_inserter(polylist));
    ksr.reconstruct(l, external_nodes, std::back_inserter(vtx), std::back_inserter(polylist));
 
    if (polylist.size() > 0) {
      non_empty = true;

      using Mesh = CGAL::Surface_mesh<Point_3>;

      // Repair the soup: removes duplicates and degenerated faces
      CGAL::Polygon_mesh_processing::repair_polygon_soup(vtx, polylist);
      // Orient the soup: fixes inconsistent normals which cause non-manifold errors
      CGAL::Polygon_mesh_processing::orient_polygon_soup(vtx, polylist);

      std::string lstr = std::to_string(CGAL::to_double(l));
      std::string filename = "polylist_" + lstr + ".ply";
      fs::path outp = outdir / filename;

      if (save_biggest_component_only) {
        // Convert polygon soup to a Surface_mesh
        Mesh full_mesh;
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(vtx, polylist, full_mesh);

        // Label connected components: each face gets an integer component ID
        Mesh::Property_map<Mesh::Face_index, std::size_t> comp_id =
          full_mesh.add_property_map<Mesh::Face_index, std::size_t>("f:component", 0).first;
        std::size_t num_components = CGAL::Polygon_mesh_processing::connected_components(full_mesh, comp_id);

        std::cout << "Lambda " << CGAL::to_double(l)
                  << ": " << num_components << " component(s)" << std::endl;

        // Count faces per component, find the largest
        std::vector<std::size_t> face_count(num_components, 0);
        for (Mesh::Face_index f : full_mesh.faces())
          ++face_count[comp_id[f]];

        std::size_t best = std::max_element(face_count.begin(), face_count.end()) - face_count.begin();
        std::cout << "  Largest component: " << best
                  << " (" << face_count[best] << " faces)" << std::endl;

        // Build sub-mesh for the largest component only
        Mesh sub_mesh;
        std::unordered_map<Mesh::Vertex_index, Mesh::Vertex_index> vmap;

        for (Mesh::Face_index f : full_mesh.faces()) {
          if (comp_id[f] != best) continue;

          std::vector<Mesh::Vertex_index> new_verts;
          for (Mesh::Vertex_index v :
               CGAL::vertices_around_face(full_mesh.halfedge(f), full_mesh)) {
            auto it = vmap.find(v);
            if (it == vmap.end()) {
              Mesh::Vertex_index nv = sub_mesh.add_vertex(full_mesh.point(v));
              vmap[v] = nv;
              new_verts.push_back(nv);
            } else {
              new_verts.push_back(it->second);
            }
          }
          sub_mesh.add_face(new_verts);
        }

        bool success = CGAL::IO::write_PLY(outp.string(), sub_mesh);
        if (success)
          std::cout << "  Wrote largest component: " << outp << std::endl;
        else
          std::cout << "  Failed to write: " << outp << std::endl;

      } else {
        // Save the full mesh (all components) as a polygon soup
        std::cout << "Lambda " << CGAL::to_double(l) << ": saving full soup" << std::endl;
        bool success = CGAL::IO::write_polygon_soup(outp.string(), vtx, polylist);
        if (success)
          std::cout << "  Wrote: " << outp << std::endl;
        else
          std::cout << "  Failed to write: " << outp << std::endl;
      }
    }
  }

  return (non_empty) ? EXIT_SUCCESS : EXIT_FAILURE;
}